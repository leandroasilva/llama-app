mod engine;
mod hf;
mod paths;
mod server;
mod settings;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

use gtk::glib;
use gtk::prelude::*;
use gtk4 as gtk;
use llama_core::engine::binaries;

use crate::engine::Engine;
use crate::hf::{ScanResult, UiModel};
use crate::server::ServerHandle;
use crate::settings::{PerfSettings, Settings};

const APP_ID: &str = "app.llama.LlamaGtk";
const BACKENDS: [&str; 2] = ["CPU", "Vulkan"];
const KV_TYPES: [&str; 7] = ["f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"];
const BATCHES: [i64; 5] = [128, 256, 512, 1024, 2048];
const UBATCHES: [i64; 6] = [64, 128, 256, 512, 1024, 2048];
const POLICIES: [&str; 2] = ["Largest that fits", "Smallest (4K)"];

enum Msg {
    Progress(u64, u64, String),
    Status(String),
    Engine(Result<String, String>),
    Models(Result<ScanResult, String>),
    Download(Result<ScanResult, String>),
}

struct State {
    settings: Settings,
    engine: Option<Engine>,
    server: Option<ServerHandle>,
    models: Vec<UiModel>,
    selected: Option<String>,
    busy: bool,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,

    window: gtk::ApplicationWindow,
    engine_label: gtk::Label,
    status: gtk::Label,
    start_btn: gtk::Button,
    stop_btn: gtk::Button,
    web_btn: gtk::Button,
    refresh_btn: gtk::Button,
    install_btn: gtk::Button,
    progress: gtk::ProgressBar,
    progress_label: gtk::Label,
    model_list: gtk::ListBox,
    detail_box: gtk::Box,

    repo_entry: gtk::Entry,
    quant_entry: gtk::Entry,
    download_btn: gtk::Button,

    port_spin: gtk::SpinButton,
    bind_switch: gtk::Switch,
    hf_entry: gtk::Entry,
    token_entry: gtk::Entry,
    api_key_entry: gtk::Entry,
    copy_key_btn: gtk::Button,
    backend_drop: gtk::DropDown,
    sleep_spin: gtk::SpinButton,
    agent_switch: gtk::Switch,
    extra_entry: gtk::Entry,

    flash_switch: gtk::Switch,
    kvk_drop: gtk::DropDown,
    kvv_drop: gtk::DropDown,
    parallel_spin: gtk::SpinButton,
    batch_drop: gtk::DropDown,
    ubatch_drop: gtk::DropDown,
    nooff_switch: gtk::Switch,
    policy_drop: gtk::DropDown,
}

impl State {
    fn save_settings(&mut self) {
        if let Err(e) = self.settings.save() {
            self.set_status(&format!("Could not save settings: {e}"));
        }
    }

    fn set_status(&mut self, text: &str) {
        self.status.set_text(text);
    }

    fn set_progress(&self, done: u64, total: u64, label: &str) {
        self.progress_label.set_text(label);
        if total > 0 {
            self.progress
                .set_fraction((done as f64 / total as f64).min(1.0));
        } else {
            self.progress.pulse();
        }
        self.progress.set_visible(true);
    }

    fn clear_progress(&self) {
        self.progress.set_visible(false);
        self.progress_label.set_text("");
    }

    fn sync_from_widgets(&mut self) {
        self.settings.server_port = self.port_spin.value_as_int() as i64;
        self.settings.bind_all = self.bind_switch.is_active();
        self.settings.hf_cache = self.hf_entry.text().to_string();
        self.settings.hf_token = self.token_entry.text().to_string();
        self.settings.backend = backend_value(&self.backend_drop);
        let sleep = self.sleep_spin.value_as_int() as i64;
        self.settings.sleep_idle_seconds = (sleep > 0).then_some(sleep);
        self.settings.agent_mode = self.agent_switch.is_active();
        self.settings.extra_args = self.extra_entry.text().to_string();
        if self.settings.bind_all && self.settings.server_api_key.is_empty() {
            self.settings.ensure_api_key();
        }
        self.api_key_entry.set_text(&self.settings.server_api_key);
        self.api_key_entry.set_visible(self.settings.bind_all);
        self.copy_key_btn.set_visible(self.settings.bind_all);
        self.settings.context_policy = if self.policy_drop.selected() == 0 {
            "largestFitting".into()
        } else {
            "minimum".into()
        };
        self.settings.performance = PerfSettings {
            flash_attention: self.flash_switch.is_active(),
            kv_cache_type_k: KV_TYPES[self.kvk_drop.selected().min(6) as usize].to_string(),
            kv_cache_type_v: KV_TYPES[self.kvv_drop.selected().min(6) as usize].to_string(),
            parallel: self.parallel_spin.value_as_int() as i64,
            batch_size: BATCHES[self.batch_drop.selected().min(4) as usize],
            ubatch_size: UBATCHES[self.ubatch_drop.selected().min(5) as usize],
            no_kv_offload: self.nooff_switch.is_active(),
        }
        .normalized();
    }

    fn refresh_engine_label(&self) {
        self.install_btn.set_label(if self.engine.is_some() {
            "Reinstall engine"
        } else {
            "Install engine"
        });
        match &self.engine {
            Some(e) => {
                let version = e
                    .version
                    .map(|v| v.tag())
                    .unwrap_or_else(|| "unknown version".into());
                self.engine_label
                    .set_text(&format!("Engine: {version}  ({})", e.origin.as_str()));
            }
            None => self.engine_label.set_text("Engine: not installed"),
        }
    }

    fn refresh_server_buttons(&self) {
        let running = self.server.is_some();
        self.start_btn.set_sensitive(!running && !self.busy);
        self.stop_btn.set_sensitive(running);
        self.web_btn.set_sensitive(running);
        self.refresh_btn.set_sensitive(!self.busy);
        self.download_btn.set_sensitive(!self.busy);
    }

    fn selected_model(&self) -> Option<&UiModel> {
        let id = self.selected.as_ref()?;
        self.models.iter().find(|m| &m.id == id)
    }
}

fn main() -> glib::ExitCode {
    paths::ensure_dirs();
    let app = gtk::Application::builder().application_id(APP_ID).build();
    app.connect_activate(|app| {
        let (tx, rx) = mpsc::channel();
        build_ui(app, tx, rx);
    });
    app.run()
}

fn build_ui(app: &gtk::Application, tx: Sender<Msg>, rx: Receiver<Msg>) -> Rc<RefCell<State>> {
    let settings = Settings::load();
    let engine = engine::resolve_engine();
    let initial = hf::scan_models(&settings, engine.as_ref().map(|e| e.path.as_str()), false);
    let _ = hf::write_models_ini(&settings, &initial.models);

    let window = gtk::ApplicationWindow::builder()
        .application(app)
        .title("Llama")
        .default_width(1080)
        .default_height(780)
        .build();

    let header = gtk::HeaderBar::new();
    let title = gtk::Label::new(Some("Llama"));
    title.add_css_class("title");
    header.set_title_widget(Some(&title));
    let install_btn = gtk::Button::with_label("Install engine");
    install_btn.add_css_class("suggested-action");
    header.pack_end(&install_btn);
    window.set_titlebar(Some(&header));

    let root = gtk::Box::new(gtk::Orientation::Vertical, 0);

    let top = gtk::Box::new(gtk::Orientation::Vertical, 8);
    top.set_margin_top(12);
    top.set_margin_bottom(8);
    top.set_margin_start(12);
    top.set_margin_end(12);

    let engine_label = gtk::Label::new(None);
    engine_label.set_xalign(0.0);
    engine_label.add_css_class("dim-label");
    let status = gtk::Label::new(Some("Ready"));
    status.set_xalign(0.0);
    status.set_hexpand(true);
    status.set_ellipsize(gtk::pango::EllipsizeMode::End);

    let controls = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    let start_btn = gtk::Button::with_label("Start server");
    start_btn.add_css_class("suggested-action");
    let stop_btn = gtk::Button::with_label("Stop");
    let web_btn = gtk::Button::with_label("Open Web UI");
    let refresh_btn = gtk::Button::with_label("Refresh");
    controls.append(&start_btn);
    controls.append(&stop_btn);
    controls.append(&web_btn);
    controls.append(&refresh_btn);

    let status_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    status_row.append(&status);
    status_row.append(&controls);

    let progress = gtk::ProgressBar::new();
    progress.set_show_text(false);
    progress.set_visible(false);
    let progress_label = gtk::Label::new(None);
    progress_label.set_xalign(0.0);
    progress_label.add_css_class("dim-label");

    top.append(&engine_label);
    top.append(&status_row);
    top.append(&progress);
    top.append(&progress_label);

    let paned = gtk::Paned::new(gtk::Orientation::Horizontal);
    paned.set_position(300);
    paned.set_vexpand(true);

    let left = gtk::Box::new(gtk::Orientation::Vertical, 0);
    let left_title = gtk::Label::new(Some("Installed models"));
    left_title.set_xalign(0.0);
    left_title.set_margin_top(8);
    left_title.set_margin_bottom(6);
    left_title.set_margin_start(12);
    left_title.set_margin_end(12);
    left_title.add_css_class("heading");
    let model_scroll = gtk::ScrolledWindow::new();
    model_scroll.set_vexpand(true);
    let model_list = gtk::ListBox::new();
    model_list.set_selection_mode(gtk::SelectionMode::Single);
    model_list.add_css_class("navigation-sidebar");
    model_scroll.set_child(Some(&model_list));
    left.append(&left_title);
    left.append(&model_scroll);

    let notebook = gtk::Notebook::new();
    notebook.set_hexpand(true);
    notebook.set_vexpand(true);

    let detail_box = gtk::Box::new(gtk::Orientation::Vertical, 10);
    detail_box.set_margin_top(16);
    detail_box.set_margin_bottom(16);
    detail_box.set_margin_start(18);
    detail_box.set_margin_end(18);
    let detail_scroll = gtk::ScrolledWindow::new();
    detail_scroll.set_child(Some(&detail_box));
    notebook.append_page(&detail_scroll, Some(&gtk::Label::new(Some("Model"))));

    let downloads_box = gtk::Box::new(gtk::Orientation::Vertical, 10);
    downloads_box.set_margin_top(16);
    downloads_box.set_margin_bottom(16);
    downloads_box.set_margin_start(18);
    downloads_box.set_margin_end(18);
    let repo_entry = gtk::Entry::new();
    repo_entry.set_placeholder_text(Some("org/repo, e.g. unsloth/Qwen3.5-2B-GGUF"));
    let quant_entry = gtk::Entry::new();
    quant_entry.set_placeholder_text(Some("Quant (optional), e.g. Q4_K_M"));
    let download_btn = gtk::Button::with_label("Download model");
    download_btn.add_css_class("suggested-action");
    downloads_box.append(&gtk::Label::new(Some("Download from Hugging Face")));
    downloads_box.append(&repo_entry);
    downloads_box.append(&quant_entry);
    downloads_box.append(&download_btn);
    let download_hint = gtk::Label::new(Some(
        "Downloads go to the Hugging Face cache and appear in Installed models when complete.",
    ));
    download_hint.set_wrap(true);
    download_hint.set_xalign(0.0);
    download_hint.add_css_class("dim-label");
    downloads_box.append(&download_hint);
    let downloads_scroll = gtk::ScrolledWindow::new();
    downloads_scroll.set_child(Some(&downloads_box));
    notebook.append_page(&downloads_scroll, Some(&gtk::Label::new(Some("Downloads"))));

    let perf = settings.performance.clone();
    let server_box = gtk::Box::new(gtk::Orientation::Vertical, 12);
    server_box.set_margin_top(16);
    server_box.set_margin_bottom(16);
    server_box.set_margin_start(18);
    server_box.set_margin_end(18);

    let server_grid = gtk::Grid::new();
    server_grid.set_row_spacing(8);
    server_grid.set_column_spacing(12);
    let port_spin = gtk::SpinButton::with_range(1024.0, 65535.0, 1.0);
    port_spin.set_value(settings.server_port as f64);
    let bind_switch = gtk::Switch::new();
    bind_switch.set_active(settings.bind_all);
    bind_switch.set_halign(gtk::Align::Start);
    let hf_entry = gtk::Entry::new();
    hf_entry.set_text(&settings.hf_cache);
    hf_entry.set_hexpand(true);
    let token_entry = gtk::Entry::new();
    token_entry.set_visibility(false);
    token_entry.set_placeholder_text(Some("hf_... (stored in gtk-settings.json, mode 600)"));
    token_entry.set_text(&settings.hf_token);
    let api_key_entry = gtk::Entry::new();
    api_key_entry.set_editable(false);
    api_key_entry.set_text(&settings.server_api_key);
    api_key_entry.set_visible(settings.bind_all);
    let copy_key_btn = gtk::Button::with_label("Copy");
    copy_key_btn.set_visible(settings.bind_all);
    let backend_drop = gtk::DropDown::from_strings(&BACKENDS);
    backend_drop.set_selected(if settings.backend == "vulkan" { 1 } else { 0 });
    backend_drop.set_halign(gtk::Align::Start);
    let sleep_spin = gtk::SpinButton::with_range(0.0, 86400.0, 60.0);
    sleep_spin.set_value(settings.sleep_idle_seconds.unwrap_or(0) as f64);
    let agent_switch = gtk::Switch::new();
    agent_switch.set_active(settings.agent_mode);
    agent_switch.set_halign(gtk::Align::Start);
    let extra_entry = gtk::Entry::new();
    extra_entry.set_placeholder_text(Some("Extra llama serve arguments (optional)"));
    extra_entry.set_text(&settings.extra_args);
    extra_entry.set_hexpand(true);

    let mut row = 0;
    add_row(&server_grid, row, "Port", &port_spin);
    row += 1;
    add_row(&server_grid, row, "Expose on 0.0.0.0", &bind_switch);
    row += 1;
    add_row(&server_grid, row, "Compute backend", &backend_drop);
    row += 1;
    add_row(&server_grid, row, "Sleep idle seconds", &sleep_spin);
    row += 1;
    add_row(&server_grid, row, "Agent mode", &agent_switch);
    row += 1;
    add_row(&server_grid, row, "HF cache", &hf_entry);
    row += 1;
    add_row(&server_grid, row, "Hugging Face token", &token_entry);
    row += 1;
    add_row(&server_grid, row, "Extra args", &extra_entry);

    let save_btn = gtk::Button::with_label("Save settings");
    let server_hint = gtk::Label::new(Some(
        "Exposing on 0.0.0.0 requires an API key, generated automatically and passed through LLAMA_API_KEY.",
    ));
    server_hint.set_wrap(true);
    server_hint.set_xalign(0.0);
    server_hint.add_css_class("dim-label");
    let api_key_row = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    let api_key_label = gtk::Label::new(Some("API key"));
    api_key_label.set_width_chars(22);
    api_key_label.set_xalign(0.0);
    api_key_row.append(&api_key_label);
    api_key_row.append(&api_key_entry);
    api_key_row.append(&copy_key_btn);
    api_key_entry.set_hexpand(true);
    server_box.append(&server_grid);
    server_box.append(&api_key_row);
    server_box.append(&server_hint);
    server_box.append(&save_btn);
    let server_scroll = gtk::ScrolledWindow::new();
    server_scroll.set_child(Some(&server_box));
    notebook.append_page(&server_scroll, Some(&gtk::Label::new(Some("Server"))));

    let perf_box = gtk::Box::new(gtk::Orientation::Vertical, 12);
    perf_box.set_margin_top(16);
    perf_box.set_margin_bottom(16);
    perf_box.set_margin_start(18);
    perf_box.set_margin_end(18);
    let perf_grid = gtk::Grid::new();
    perf_grid.set_row_spacing(8);
    perf_grid.set_column_spacing(12);
    let flash_switch = gtk::Switch::new();
    flash_switch.set_active(perf.flash_attention);
    flash_switch.set_halign(gtk::Align::Start);
    let kvk_drop = gtk::DropDown::from_strings(&KV_TYPES);
    kvk_drop.set_selected(index_of(&KV_TYPES, &perf.kv_cache_type_k).unwrap_or(1));
    kvk_drop.set_halign(gtk::Align::Start);
    let kvv_drop = gtk::DropDown::from_strings(&KV_TYPES);
    kvv_drop.set_selected(index_of(&KV_TYPES, &perf.kv_cache_type_v).unwrap_or(1));
    kvv_drop.set_halign(gtk::Align::Start);
    let parallel_spin = gtk::SpinButton::with_range(1.0, 8.0, 1.0);
    parallel_spin.set_value(perf.parallel as f64);
    let batch_labels = BATCHES.map(|v| v.to_string());
    let batch_refs: Vec<&str> = batch_labels.iter().map(String::as_str).collect();
    let batch_drop = gtk::DropDown::from_strings(&batch_refs);
    batch_drop.set_selected(index_i64(&BATCHES, perf.batch_size).unwrap_or(2));
    batch_drop.set_halign(gtk::Align::Start);
    let ubatch_labels = UBATCHES.map(|v| v.to_string());
    let ubatch_refs: Vec<&str> = ubatch_labels.iter().map(String::as_str).collect();
    let ubatch_drop = gtk::DropDown::from_strings(&ubatch_refs);
    ubatch_drop.set_selected(index_i64(&UBATCHES, perf.ubatch_size).unwrap_or(2));
    ubatch_drop.set_halign(gtk::Align::Start);
    let nooff_switch = gtk::Switch::new();
    nooff_switch.set_active(perf.no_kv_offload);
    nooff_switch.set_halign(gtk::Align::Start);
    let policy_drop = gtk::DropDown::from_strings(&POLICIES);
    policy_drop.set_selected(if settings.context_policy == "minimum" {
        1
    } else {
        0
    });
    policy_drop.set_halign(gtk::Align::Start);

    let mut prow = 0;
    add_row(&perf_grid, prow, "Flash attention", &flash_switch);
    prow += 1;
    add_row(&perf_grid, prow, "KV cache K", &kvk_drop);
    prow += 1;
    add_row(&perf_grid, prow, "KV cache V", &kvv_drop);
    prow += 1;
    add_row(&perf_grid, prow, "Parallel slots", &parallel_spin);
    prow += 1;
    add_row(&perf_grid, prow, "Batch size", &batch_drop);
    prow += 1;
    add_row(&perf_grid, prow, "Ubatch size", &ubatch_drop);
    prow += 1;
    add_row(&perf_grid, prow, "No KV offload", &nooff_switch);
    prow += 1;
    add_row(&perf_grid, prow, "Default context", &policy_drop);
    let save_perf_btn = gtk::Button::with_label("Save performance settings");
    let perf_hint = gtk::Label::new(Some(
        "These values go into the [*] section of models.ini. Restart the server to apply them.",
    ));
    perf_hint.set_wrap(true);
    perf_hint.set_xalign(0.0);
    perf_hint.add_css_class("dim-label");
    perf_box.append(&perf_grid);
    perf_box.append(&perf_hint);
    perf_box.append(&save_perf_btn);
    let perf_scroll = gtk::ScrolledWindow::new();
    perf_scroll.set_child(Some(&perf_box));
    notebook.append_page(&perf_scroll, Some(&gtk::Label::new(Some("Performance"))));

    paned.set_start_child(Some(&left));
    paned.set_end_child(Some(&notebook));
    root.append(&top);
    root.append(&paned);
    window.set_child(Some(&root));

    let state = Rc::new(RefCell::new(State {
        settings,
        engine,
        server: None,
        models: initial.models,
        selected: None,
        busy: false,
        tx,
        rx,
        window: window.clone(),
        engine_label,
        status,
        start_btn: start_btn.clone(),
        stop_btn: stop_btn.clone(),
        web_btn: web_btn.clone(),
        refresh_btn: refresh_btn.clone(),
        install_btn: install_btn.clone(),
        progress,
        progress_label,
        model_list: model_list.clone(),
        detail_box,
        repo_entry: repo_entry.clone(),
        quant_entry: quant_entry.clone(),
        download_btn: download_btn.clone(),
        port_spin,
        bind_switch,
        hf_entry,
        token_entry,
        api_key_entry: api_key_entry.clone(),
        copy_key_btn: copy_key_btn.clone(),
        backend_drop,
        sleep_spin,
        agent_switch,
        extra_entry,
        flash_switch,
        kvk_drop,
        kvv_drop,
        parallel_spin,
        batch_drop,
        ubatch_drop,
        nooff_switch,
        policy_drop,
    }));

    state.borrow_mut().refresh_engine_label();
    rebuild_model_list(&state);
    state.borrow().refresh_server_buttons();

    {
        let state = Rc::clone(&state);
        model_list.connect_row_selected(move |_, row| {
            let Some(row) = row else { return };
            let index = row.index().max(0) as usize;
            let id = {
                let st = state.borrow();
                st.models.get(index).map(|m| m.id.clone())
            };
            if let Some(id) = id {
                let mut st = state.borrow_mut();
                st.selected = Some(id);
                drop(st);
                rebuild_detail(&state);
            }
        });
    }

    {
        let state = Rc::clone(&state);
        start_btn.connect_clicked(move |_| start_server(&state));
    }
    {
        let state = Rc::clone(&state);
        stop_btn.connect_clicked(move |_| stop_server(&state));
    }
    {
        let state = Rc::clone(&state);
        web_btn.connect_clicked(move |_| {
            let port = state
                .borrow()
                .server
                .as_ref()
                .map(|s| s.port)
                .unwrap_or(9931);
            server::open_web_ui(port);
        });
    }
    {
        let state = Rc::clone(&state);
        refresh_btn.connect_clicked(move |_| start_scan(&state));
    }
    {
        let state = Rc::clone(&state);
        install_btn.connect_clicked(move |_| install_engine(&state));
    }
    {
        let state = Rc::clone(&state);
        download_btn.connect_clicked(move |_| start_download(&state));
    }
    {
        let state = Rc::clone(&state);
        save_btn.connect_clicked(move |_| {
            let mut st = state.borrow_mut();
            st.sync_from_widgets();
            st.save_settings();
            let models = st.models.clone();
            let settings = st.settings.clone();
            drop(st);
            match hf::write_models_ini(&settings, &models) {
                Ok(_) => state.borrow_mut().set_status("Settings saved."),
                Err(e) => state
                    .borrow_mut()
                    .set_status(&format!("Couldn't write models.ini: {e}")),
            }
        });
    }
    {
        let state = Rc::clone(&state);
        let switch = state.borrow().bind_switch.clone();
        switch.connect_active_notify(move |_| {
            let mut st = state.borrow_mut();
            st.sync_from_widgets();
            st.save_settings();
        });
    }
    {
        let state = Rc::clone(&state);
        copy_key_btn.connect_clicked(move |_| {
            let key = {
                let mut st = state.borrow_mut();
                if st.settings.server_api_key.is_empty() {
                    st.settings.ensure_api_key();
                }
                st.settings.server_api_key.clone()
            };
            if !key.is_empty() {
                state.borrow().copy_key_btn.clipboard().set_text(&key);
                state
                    .borrow_mut()
                    .set_status("API key copied to the clipboard.");
            }
        });
    }

    {
        let state = Rc::clone(&state);
        save_perf_btn.connect_clicked(move |_| {
            let mut st = state.borrow_mut();
            st.sync_from_widgets();
            st.save_settings();
            let models = st.models.clone();
            let settings = st.settings.clone();
            drop(st);
            match hf::write_models_ini(&settings, &models) {
                Ok(_) => state
                    .borrow_mut()
                    .set_status("Performance settings saved. Restart the server to apply."),
                Err(e) => state
                    .borrow_mut()
                    .set_status(&format!("Couldn't write models.ini: {e}")),
            }
        });
    }

    {
        let state = Rc::clone(&state);
        let window = window.clone();
        window.connect_close_request(move |_| {
            let mut st = state.borrow_mut();
            if let Some(s) = st.server.as_mut() {
                s.stop();
            }
            let _ = st.settings.save();
            glib::Propagation::Proceed
        });
    }

    {
        let state = Rc::clone(&state);
        glib::timeout_add_local(Duration::from_millis(100), move || {
            let msgs: Vec<Msg> = {
                let st = state.borrow();
                st.rx.try_iter().collect()
            };
            for msg in msgs {
                handle_msg(&state, msg);
            }
            glib::ControlFlow::Continue
        });
    }

    {
        let state = Rc::clone(&state);
        glib::timeout_add_seconds_local(1, move || {
            let exited = {
                let mut st = state.borrow_mut();
                let Some(server) = st.server.as_mut() else {
                    return glib::ControlFlow::Continue;
                };
                server.try_wait()
            };
            if let Some(status) = exited {
                let mut st = state.borrow_mut();
                st.server = None;
                st.set_status(&format!("Server stopped ({status})."));
                st.refresh_server_buttons();
            }
            glib::ControlFlow::Continue
        });
    }

    start_scan(&state);
    state
}

fn add_row(grid: &gtk::Grid, row: i32, label: &str, widget: &impl IsA<gtk::Widget>) {
    let l = gtk::Label::new(Some(label));
    l.set_xalign(0.0);
    l.set_width_chars(22);
    grid.attach(&l, 0, row, 1, 1);
    grid.attach(widget, 1, row, 1, 1);
}

fn backend_value(drop: &gtk::DropDown) -> String {
    if drop.selected() == 1 {
        "vulkan".into()
    } else {
        "cpu".into()
    }
}

fn index_of(values: &[&str], needle: &str) -> Option<u32> {
    values.iter().position(|v| *v == needle).map(|i| i as u32)
}

fn index_i64(values: &[i64], needle: i64) -> Option<u32> {
    values.iter().position(|v| *v == needle).map(|i| i as u32)
}

fn rebuild_model_list(state: &Rc<RefCell<State>>) {
    let (list, models, selected) = {
        let st = state.borrow();
        (
            st.model_list.clone(),
            st.models.clone(),
            st.selected.clone(),
        )
    };
    while let Some(child) = list.first_child() {
        list.remove(&child);
    }
    if models.is_empty() {
        let row = gtk::ListBoxRow::new();
        row.set_activatable(false);
        row.set_selectable(false);
        let label = gtk::Label::new(Some("No models installed"));
        label.set_margin_top(12);
        label.set_margin_bottom(12);
        label.set_margin_start(12);
        label.set_margin_end(12);
        label.set_xalign(0.0);
        label.add_css_class("dim-label");
        row.set_child(Some(&label));
        list.append(&row);
        return;
    }
    let mut select_index = None;
    for (index, model) in models.iter().enumerate() {
        let row = gtk::ListBoxRow::new();
        let b = gtk::Box::new(gtk::Orientation::Vertical, 2);
        b.set_margin_top(8);
        b.set_margin_bottom(8);
        b.set_margin_start(10);
        b.set_margin_end(10);
        let name = gtk::Label::new(Some(&model.display_name));
        name.set_xalign(0.0);
        name.set_ellipsize(gtk::pango::EllipsizeMode::End);
        let sub = format!(
            "{} · {}",
            model
                .effective_tier
                .map(tier_label)
                .unwrap_or_else(|| "no context profile".into()),
            human_size(model.file_size)
        );
        let meta = gtk::Label::new(Some(&sub));
        meta.set_xalign(0.0);
        meta.add_css_class("dim-label");
        b.append(&name);
        b.append(&meta);
        row.set_child(Some(&b));
        list.append(&row);
        if selected.as_deref() == Some(model.id.as_str()) {
            select_index = Some(index as i32);
        }
    }
    if let Some(index) = select_index {
        if let Some(row) = list.row_at_index(index) {
            list.select_row(Some(&row));
        }
    }
}

fn rebuild_detail(state: &Rc<RefCell<State>>) {
    let (container, model, policy) = {
        let st = state.borrow();
        (
            st.detail_box.clone(),
            st.selected_model().cloned(),
            st.settings.context_policy.clone(),
        )
    };
    while let Some(child) = container.first_child() {
        container.remove(&child);
    }
    let Some(model) = model else {
        let label = gtk::Label::new(Some("Select a model to see its settings."));
        label.set_xalign(0.0);
        label.add_css_class("dim-label");
        container.append(&label);
        return;
    };

    let name = gtk::Label::new(Some(&model.id));
    name.set_xalign(0.0);
    name.set_wrap(true);
    name.add_css_class("title-2");
    container.append(&name);

    let tags = if model.is_decision_model {
        "decision model"
    } else {
        "chat model"
    };
    let meta = gtk::Label::new(Some(&format!(
        "{tags} · {} · context window {} · {}",
        human_size(model.file_size),
        tier_label(model.ctx_window),
        model.model_path
    )));
    meta.set_xalign(0.0);
    meta.set_wrap(true);
    meta.add_css_class("dim-label");
    container.append(&meta);

    let ctx_label = gtk::Label::new(Some("Context length"));
    ctx_label.set_xalign(0.0);
    ctx_label.set_margin_top(12);
    container.append(&ctx_label);

    let auto_label = if policy == "minimum" {
        "Auto (smallest, 4K)"
    } else {
        "Auto (largest that fits)"
    };
    let mut labels: Vec<String> = vec![auto_label.into()];
    let mut values: Vec<Option<i64>> = vec![None];
    for tier in &model.supported_tiers {
        labels.push(tier_label(*tier));
        values.push(Some(*tier));
    }
    let refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    let dropdown = gtk::DropDown::from_strings(&refs);
    let current = model
        .selected_tier
        .and_then(|t| values.iter().position(|v| *v == Some(t)))
        .unwrap_or(0);
    dropdown.set_selected(current as u32);
    dropdown.set_halign(gtk::Align::Start);
    container.append(&dropdown);

    let note = gtk::Label::new(Some(
        "Tiers come from the measured memory profile. Unsupported tiers are hidden.",
    ));
    note.set_xalign(0.0);
    note.add_css_class("dim-label");
    note.set_wrap(true);
    container.append(&note);

    let delete_btn = gtk::Button::with_label("Delete model files");
    delete_btn.add_css_class("destructive-action");
    delete_btn.set_halign(gtk::Align::Start);
    delete_btn.set_margin_top(16);
    container.append(&delete_btn);

    {
        let state = Rc::clone(state);
        let model_id = model.id.clone();
        let model_values = values.clone();
        dropdown.connect_selected_notify(move |selector| {
            let idx = selector.selected() as usize;
            let chosen = model_values.get(idx).copied().flatten();
            {
                let mut st = state.borrow_mut();
                match chosen {
                    Some(tier) => {
                        st.settings.selected_tiers.insert(model_id.clone(), tier);
                    }
                    None => {
                        st.settings.selected_tiers.remove(&model_id);
                    }
                }
                let prefer_largest = st.settings.context_policy == "largestFitting";
                if let Some(model) = st.models.iter_mut().find(|m| m.id == model_id) {
                    model.selected_tier = chosen;
                    model.effective_tier = chosen.or_else(|| {
                        if prefer_largest {
                            model.supported_tiers.last().copied()
                        } else {
                            model.supported_tiers.first().copied()
                        }
                    });
                }
                st.save_settings();
                let models = st.models.clone();
                let settings = st.settings.clone();
                drop(st);
                let _ = hf::write_models_ini(&settings, &models);
            }
            state
                .borrow_mut()
                .set_status("Context setting saved. It applies on the next server start.");
        });
    }

    {
        let state = Rc::clone(state);
        let model_id = model.id.clone();
        let display = model.display_name.clone();
        delete_btn.connect_clicked(move |_| {
            let dialog = gtk::MessageDialog::builder()
                .transient_for(&state.borrow().window)
                .modal(true)
                .message_type(gtk::MessageType::Question)
                .buttons(gtk::ButtonsType::YesNo)
                .text(format!("Delete {display}?"))
                .secondary_text("This removes the model files from the Hugging Face cache.")
                .build();
            let state = Rc::clone(&state);
            let model_id = model_id.clone();
            dialog.connect_response(move |dialog, response| {
                if response == gtk::ResponseType::Yes {
                    let model = {
                        let st = state.borrow();
                        st.models.iter().find(|m| m.id == model_id).cloned()
                    };
                    if let Some(model) = model {
                        let cache = state.borrow().settings.hf_cache_path();
                        match hf::delete_model(&cache, &model) {
                            Ok(()) => {
                                let mut st = state.borrow_mut();
                                st.selected = None;
                                st.set_status("Model deleted.");
                                drop(st);
                                start_scan(&state);
                            }
                            Err(e) => state
                                .borrow_mut()
                                .set_status(&format!("Couldn't delete model: {e}")),
                        }
                    }
                }
                dialog.close();
            });
            dialog.present();
        });
    }
}

fn tier_label(raw: i64) -> String {
    format!("{} ctx", raw / 1024)
}

fn human_size(bytes: i64) -> String {
    let gb = bytes as f64 / 1_073_741_824.0;
    if gb >= 1.0 {
        format!("{gb:.1} GB")
    } else {
        format!("{:.0} MB", bytes as f64 / 1_048_576.0)
    }
}

fn start_server(state: &Rc<RefCell<State>>) {
    let (settings, engine_path, has_models) = {
        let mut st = state.borrow_mut();
        st.sync_from_widgets();
        st.save_settings();
        let models = st.models.clone();
        let settings = st.settings.clone();
        let _ = hf::write_models_ini(&settings, &models);
        (
            st.settings.clone(),
            st.engine.as_ref().map(|e| e.path.clone()),
            !st.models.is_empty(),
        )
    };
    if !has_models {
        state
            .borrow_mut()
            .set_status("Download a model before starting the server.");
        return;
    }
    let Some(engine_path) = engine_path else {
        state
            .borrow_mut()
            .set_status("Install the llama.cpp engine first.");
        return;
    };
    if server::port_in_use(settings.server_port) {
        state.borrow_mut().set_status(&format!(
            "Port {} is already in use. Pick another port in Server settings.",
            settings.server_port
        ));
        return;
    }
    let mut settings = settings;
    match server::start(&mut settings, &engine_path) {
        Ok(handle) => {
            let mut st = state.borrow_mut();
            st.settings = settings;
            let port = st.settings.server_port;
            st.server = Some(handle);
            st.set_status(&format!("Server running on http://localhost:{port}."));
            st.refresh_server_buttons();
        }
        Err(e) => state.borrow_mut().set_status(&e),
    }
}

fn stop_server(state: &Rc<RefCell<State>>) {
    let mut st = state.borrow_mut();
    if let Some(mut server) = st.server.take() {
        server.stop();
        st.set_status("Server stopped.");
        st.refresh_server_buttons();
    }
}

fn start_scan(state: &Rc<RefCell<State>>) {
    {
        let mut st = state.borrow_mut();
        st.sync_from_widgets();
        st.save_settings();
        if st.busy {
            return;
        }
        st.busy = true;
        st.refresh_server_buttons();
        st.set_status("Scanning models and memory profiles…");
        st.set_progress(0, 0, "Scanning models…");
    }
    let (tx, settings, engine_path) = {
        let st = state.borrow();
        (
            st.tx.clone(),
            st.settings.clone(),
            st.engine.as_ref().map(|e| e.path.clone()),
        )
    };
    std::thread::spawn(move || {
        let result = hf::scan_models(&settings, engine_path.as_deref(), true);
        let _ = hf::write_models_ini(&settings, &result.models);
        let _ = tx.send(Msg::Models(Ok(result)));
    });
}

fn install_engine(state: &Rc<RefCell<State>>) {
    {
        let mut st = state.borrow_mut();
        if st.busy {
            return;
        }
        st.sync_from_widgets();
        st.busy = true;
        st.refresh_server_buttons();
        st.set_progress(0, 0, "Preparing engine download…");
    }
    let (tx, backend, version) = {
        let st = state.borrow();
        (
            st.tx.clone(),
            st.settings.backend.clone(),
            binaries::target_version().tag(),
        )
    };
    std::thread::spawn(move || {
        let txp = tx.clone();
        let progress = move |done: u64, total: u64, label: &str| {
            let _ = txp.send(Msg::Progress(done, total, label.to_string()));
        };
        let result = engine::install(&version, &backend, &progress);
        let _ = tx.send(Msg::Engine(result));
    });
}

fn start_download(state: &Rc<RefCell<State>>) {
    let (tx, settings, repo, quant) = {
        let mut st = state.borrow_mut();
        st.sync_from_widgets();
        st.save_settings();
        if st.busy {
            return;
        }
        let repo = st.repo_entry.text().to_string();
        let quant = st.quant_entry.text().trim().to_string();
        st.busy = true;
        st.refresh_server_buttons();
        st.set_status("Resolving model…");
        st.set_progress(0, 0, "Resolving model…");
        (
            st.tx.clone(),
            st.settings.clone(),
            repo,
            (!quant.is_empty()).then_some(quant),
        )
    };
    std::thread::spawn(move || {
        let result = (|| -> Result<ScanResult, String> {
            let repo = hf::normalize_repo(&repo)?;
            let budget = (paths::physical_memory_mb() - 4096).max(0) as f64;
            let resolved = hf::resolve(&repo, quant.as_deref(), budget, settings.token())?;
            let txp = tx.clone();
            let progress = move |done: u64, total: u64, label: &str| {
                let _ = txp.send(Msg::Progress(done, total, label.to_string()));
            };
            let _ = tx.send(Msg::Status(format!(
                "Downloading {} ({})",
                resolved.model_id,
                human_size(resolved.approximate_bytes)
            )));
            hf::download_resolved(&resolved, &settings, &progress)?;
            let result = hf::scan_models(&settings, None, false);
            let _ = hf::write_models_ini(&settings, &result.models);
            Ok(result)
        })();
        let _ = tx.send(Msg::Download(result));
    });
}

fn handle_msg(state: &Rc<RefCell<State>>, msg: Msg) {
    match msg {
        Msg::Progress(done, total, label) => {
            state.borrow().set_progress(done, total, &label);
        }
        Msg::Status(text) => {
            state.borrow_mut().set_status(&text);
        }
        Msg::Engine(result) => {
            let mut st = state.borrow_mut();
            st.busy = false;
            match result {
                Ok(path) => {
                    st.engine = engine::resolve_engine().or(Some(Engine {
                        path,
                        origin: binaries::Origin::Managed,
                        version: binaries::target_version().into(),
                    }));
                    st.refresh_engine_label();
                    st.set_status("Engine installed.");
                }
                Err(e) => st.set_status(&format!("Engine install failed: {e}")),
            }
            st.clear_progress();
            st.refresh_server_buttons();
            drop(st);
            start_scan(state);
        }
        Msg::Models(result) => {
            let mut st = state.borrow_mut();
            st.busy = false;
            st.clear_progress();
            match result {
                Ok(result) => {
                    let profiled = result.profiled;
                    st.models = result.models;
                    if st
                        .selected
                        .as_ref()
                        .is_some_and(|id| !st.models.iter().any(|m| &m.id == id))
                    {
                        st.selected = None;
                    }
                    let suffix = if profiled > 0 {
                        format!(" ({profiled} memory profile(s) updated)")
                    } else {
                        String::new()
                    };
                    let count = st.models.len();
                    st.set_status(&format!("{count} model(s) installed{suffix}."));
                }
                Err(e) => st.set_status(&format!("Scan failed: {e}")),
            }
            st.refresh_server_buttons();
            drop(st);
            rebuild_model_list(state);
            rebuild_detail(state);
        }
        Msg::Download(result) => {
            let mut st = state.borrow_mut();
            st.busy = false;
            st.clear_progress();
            match result {
                Ok(result) => {
                    st.models = result.models;
                    st.set_status("Model downloaded.");
                    st.refresh_server_buttons();
                    drop(st);
                    rebuild_model_list(state);
                    rebuild_detail(state);
                    start_scan(state);
                }
                Err(e) => {
                    st.set_status(&format!("Download failed: {e}"));
                    st.refresh_server_buttons();
                }
            }
        }
    }
}

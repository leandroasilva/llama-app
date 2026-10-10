//! Fase 3: network + remote selection.
//!
//! Pure selection (`hf_repo_resolver`, `catalog`) and the HF header/response
//! rules (`hf_request`, `response`) are transport-independent. `client` is the
//! blocking transport, and `hf_api` wires the two together for the real fetch.

pub mod catalog;
pub mod client;
pub mod hf_api;
pub mod hf_repo_resolver;
pub mod hf_request;
pub mod response;

pub use catalog::{Family, Suggestion};
pub use client::{HttpClient, NetError, ReqwestClient};
pub use hf_api::{fetch_catalog, fetch_file_metadata, fetch_siblings};
pub use hf_repo_resolver::{ResolveError, Resolved, Sibling};
pub use response::{FileMetadata, HttpResponse};

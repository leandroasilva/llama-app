#ifndef LLAMA_CORE_FFI_H
#define LLAMA_CORE_FFI_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Initializes the core with a JSON `CoreConfig` snapshot pushed by the host.
 * Passing NULL or an empty string means "no config yet". Returns 0 on success.
 *
 * # Safety
 * `config_json` must be NULL or point to a NUL-terminated UTF-8 string that is
 * valid for the duration of the call.
 */
int llama_core_init(const char *config_json);

/**
 * Replaces the `CoreConfig` snapshot (called on every settings change in the
 * host). Returns 0 on success.
 *
 * # Safety
 * Same requirements as [`llama_core_init`].
 */
int llama_core_set_config(const char *config_json);

/**
 * Registers (or clears, with NULL) the event callback.
 *
 * # Safety
 * `cb` must be NULL or a valid C function pointer that stays callable for as
 * long as the core may emit events. The core holds it until replaced.
 */
void llama_core_set_event_cb(void (*cb)(const char*));

/**
 * Dispatch entry point. `method` selects the operation; `args_json` is a JSON
 * object. Returns a heap JSON string owned by the core, or NULL on panic.
 *
 * The caller must free the result with [`llama_core_free`].
 *
 * # Safety
 * Both pointers must be NULL or point to NUL-terminated UTF-8 strings valid
 * for the duration of the call.
 */
char *llama_core_call(const char *method, const char *args_json);

/**
 * Frees a string previously returned by [`llama_core_call`].
 *
 * # Safety
 * `s` must be NULL or a pointer previously returned by [`llama_core_call`] and
 * not yet freed; it must not be used after this call.
 */
void llama_core_free(char *s);

/**
 * Appends `len` raw bytes from `data` to the streaming `.partial` writer `handle`.
 *
 * Deliberately not on the JSON bridge: download bytes are arbitrary binary
 * (GGUF is not UTF-8), so they cannot ride the `char *args_json` contract that
 * every other `partial.*` op uses. The JSON ops (`partial.open`,
 * `partial.digest`, `partial.restart`, ...) stay in place for lifecycle and
 * reads; this is the one write-side escape hatch.
 *
 * Returns 0 on success, -2 when no writer is open at `handle`, -3 when the
 * write failed, and -1 on a null `data` with a non-zero `len`.
 *
 * # Safety
 * `data` must be NULL only when `len` is 0, else a pointer to at least `len`
 * readable bytes. `handle` must come from a live `partial.open`.
 */
int llama_core_partial_append(uint64_t handle, const uint8_t *data, size_t len);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* LLAMA_CORE_FFI_H */

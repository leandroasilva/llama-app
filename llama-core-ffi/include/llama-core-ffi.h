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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* LLAMA_CORE_FFI_H */

//! Utilities for interacting with Web APIs.

pub mod browser;

#[cfg(target_arch = "wasm32")]
mod error;
// Optional: `fs::File` holds a `*mut u8`, so its `re_async::AsyncReadAt` impl
// is not `Send`/`Sync` and does not compile for multi-threaded (atomics) wasm.
// `error` stays unconditional -- `browser` uses it too.
#[cfg(all(target_arch = "wasm32", feature = "fs"))]
pub mod fs;

#[cfg(target_arch = "wasm32")]
pub use error::Error;

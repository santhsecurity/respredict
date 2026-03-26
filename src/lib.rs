//! Response prediction: predict HTTP response characteristics before sending.
//! Skip redundant requests by predicting status codes, content types, and approximate sizes.

#![warn(missing_docs)]
#![warn(clippy::pedantic)]

mod model;
mod predictor;
mod types;

pub use model::ResponsePredictor;
pub use predictor::matches_prediction;
pub use types::{HeaderMap, ObservedResponse, Prediction, RequestContext, SkipPolicy};

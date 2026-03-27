//! Response prediction: predict HTTP response characteristics before sending.
//! Skip redundant requests by predicting status codes, content types, and approximate sizes.
//!
//! # Overview
//!
//! `respredict` provides a lightweight, statistical approach to predicting HTTP
//! responses based on observed traffic patterns. It learns from previous
//! request-response pairs to forecast characteristics of future responses,
//! enabling optimization strategies like request skipping when predictions
//! are sufficiently confident.
//!
//! # Quick Start
//!
//! ```
//! use respredict::{ResponsePredictor, RequestContext, ObservedResponse, SkipPolicy};
//!
//! // Create a predictor
//! let mut predictor = ResponsePredictor::new();
//!
//! // Train with observed responses
//! let request = RequestContext::new("https://api.example.com/users/1", "GET");
//! let response = ObservedResponse::new(200, Some("application/json"), 256);
//! predictor.train(&request, &response);
//!
//! // Predict future responses
//! if let Some(prediction) = predictor.predict(&request) {
//!     println!("Predicted status: {:?}", prediction.status);
//!     println!("Confidence: {}", prediction.confidence);
//! }
//!
//! // Check if a request can be skipped based on policy
//! let policy = SkipPolicy::default();
//! if predictor.should_skip(&request, &policy) {
//!     println!("Request can be skipped with high confidence");
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::pedantic)]

mod model;
mod predictor;
mod types;

pub use model::ResponsePredictor;
pub use predictor::matches_prediction;
pub use types::{HeaderMap, ObservedResponse, Prediction, RequestContext, SkipPolicy};

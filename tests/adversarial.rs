use respredict::{
    matches_prediction, ObservedResponse, Prediction, RequestContext, ResponsePredictor, SkipPolicy,
};
use std::sync::Arc;
use std::thread;

#[test]
fn empty_and_invalid_inputs_fail_cleanly() {
    let predictor = ResponsePredictor::new();
    assert!(predictor.predict(&RequestContext::new("", "")).is_none());
    assert!(!predictor.should_skip(
        &RequestContext::new("not-a-url", "GET"),
        &SkipPolicy::default()
    ));
}

#[test]
fn null_bytes_and_unicode_urls_are_rejected_without_panicking() {
    let predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/\0/🧪?q=☃️", "get");
    assert!(predictor.predict(&request).is_none());
}

#[test]
fn extremely_long_urls_and_headers_are_handled() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new(format!("https://example.com/{}", "a".repeat(16_384)), "GET")
        .with_header("Accept", "text/html")
        .with_header("X-Long", "b".repeat(16_384));

    predictor.train(
        &request,
        &ObservedResponse::new(200, Some("text/html"), usize::MAX),
    );
    let prediction = predictor
        .predict(&request)
        .expect("prediction should exist");
    assert_eq!(prediction.status, Some(200));
    assert_eq!(prediction.approximate_size, Some(usize::MAX));
}

#[test]
fn malformed_training_data_is_ignored_not_panicked() {
    let mut predictor = ResponsePredictor::new();
    predictor.train(
        &RequestContext::new("this is not a url", "POST"),
        &ObservedResponse::new(0, None, 0),
    );
    assert!(predictor
        .predict(&RequestContext::new("https://example.com/test", "POST"))
        .is_none());
}

#[test]
fn negative_tolerance_is_treated_as_zero() {
    let prediction = Prediction {
        status: Some(200),
        content_type: Some("text/plain".into()),
        approximate_size: Some(100),
        confidence: 1.0,
        samples: 1,
    };

    assert!(matches_prediction(
        &prediction,
        &ObservedResponse::new(200, Some("text/plain"), 100),
        -1.0,
    ));
    assert!(!matches_prediction(
        &prediction,
        &ObservedResponse::new(200, Some("text/plain"), 101),
        -1.0,
    ));
}

#[test]
fn zero_and_max_boundaries_do_not_overflow() {
    let prediction = Prediction {
        status: Some(u16::MAX),
        content_type: None,
        approximate_size: Some(usize::MAX),
        confidence: 0.0,
        samples: 0,
    };
    assert!(matches_prediction(
        &prediction,
        &ObservedResponse::new(u16::MAX, None, usize::MAX),
        1.0,
    ));
}

#[test]
fn every_public_function_handles_bad_input_gracefully() {
    let mut predictor = ResponsePredictor::new();
    predictor.train_batch(vec![(
        RequestContext::new("notaurl", "TRACE"),
        ObservedResponse::new(999, Some("☃️/weird"), 0),
    )]);

    assert!(predictor
        .predict(&RequestContext::new("notaurl", "TRACE"))
        .is_none());
    assert!(!predictor.should_skip(
        &RequestContext::new("", ""),
        &SkipPolicy {
            min_confidence: f32::NAN,
            min_samples: 0,
            max_size_spread_ratio: -1.0,
        },
    ));
}

#[test]
fn concurrent_prediction_reads_are_safe() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ResponsePredictor>();

    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/data", "GET")
        .with_header("Accept", "application/json");
    for _ in 0..4 {
        predictor.train(
            &request,
            &ObservedResponse::new(200, Some("application/json"), 128),
        );
    }

    let predictor = Arc::new(predictor);
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let predictor = Arc::clone(&predictor);
            let request = request.clone();
            thread::spawn(move || {
                let prediction = predictor
                    .predict(&request)
                    .expect("prediction should exist");
                assert_eq!(prediction.status, Some(200));
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread failed");
    }
}

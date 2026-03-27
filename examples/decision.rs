use respredict::{
    matches_prediction, ObservedResponse, Prediction, RequestContext, ResponsePredictor,
};

fn main() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/data", "GET")
        .with_header("Accept", "application/json");

    for size in [1024, 1028, 1032, 1029] {
        predictor.train(
            &request,
            &ObservedResponse::new(200, Some("application/json"), size),
        );
    }

    let prediction = predictor.predict(&request).expect("prediction");
    let observed = ObservedResponse::new(200, Some("application/json"), 1030);

    let matches = matches_prediction(&prediction, &observed, 0.05);
    let fallback = matches_prediction(
        &Prediction {
            status: Some(404),
            content_type: Some("application/json".to_string()),
            approximate_size: Some(1024),
            confidence: prediction.confidence,
            samples: prediction.samples,
        },
        &observed,
        0.05,
    );

    println!("prediction={prediction:?}");
    println!("observed_matches={matches}");
    println!("forced_status_mismatch={fallback}");
}

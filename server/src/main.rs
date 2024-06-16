use actix_web::{get, post, App, HttpResponse, HttpServer, Responder};


#[get("/")]
async fn health() -> impl Responder {
    HttpResponse::Ok().body("health")
}

#[post("/predict")]
async fn predict(req_body: String) -> impl Responder {
    HttpResponse::Ok().body("xx: {req_body}")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Hello, world!");
    let addr = ("127.0.0.1", 8080);
    HttpServer::new(|| {
        App::new()
        .service(health)
        .service(predict)
    })
    .bind(addr)?
    .run()
    .await
}

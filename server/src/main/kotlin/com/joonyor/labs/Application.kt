package com.joonyor.labs

import io.ktor.http.*
import io.ktor.server.application.*
import io.ktor.server.http.content.*
import io.ktor.server.response.*
import io.ktor.server.routing.*

fun main(args: Array<String>) = io.ktor.server.netty.EngineMain.main(args)

fun Application.module() {

    // routing
    routing {
        staticResources("/", "/web")
        get("/") {
            call.respond(HttpStatusCode.OK,"Hello World!")
        }
    }
    emailConfiguration()
}

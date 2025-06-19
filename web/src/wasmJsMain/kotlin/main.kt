package com.joonyor.labs

import kotlin.js.*
import kotlinx.browser.*

@JsModule("htmx.org")
external object htmx

fun main() {
    document.body?.apply {
        // do work
    }
}

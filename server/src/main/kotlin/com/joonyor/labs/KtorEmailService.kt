package com.joonyor.labs

import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.plugins.requestvalidation.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import io.ktor.util.logging.*
import kotlinx.coroutines.*
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import redis.clients.jedis.Jedis
import redis.clients.jedis.JedisPool
import redis.clients.jedis.JedisPoolConfig
import java.time.Duration
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.PriorityBlockingQueue
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import javax.mail.*
import javax.mail.internet.InternetAddress
import javax.mail.internet.MimeMessage

fun Application.emailConfiguration() {
    // setup service
    val emailService = KtorEmailService(
        emailConfig = EmailConfig(
            from = environment.config.property("email.from").getString(),
            password = environment.config.property("email.password").getString(),
            host = environment.config.property("email.host").getString(),
            port = environment.config.property("email.port").getString().toInt()
        )
    )

    // Choose queue type from configuration
    val queueTypeName = environment.config.propertyOrNull("email.queue.type")?.getString() ?: "IN_MEMORY"
    val queueType = EmailQueueType.valueOf(queueTypeName)

    // Create queue config map
    val queueConfig = mapOf(
        "redisHost" to (environment.config.propertyOrNull("redis.host")?.getString() ?: "localhost"),
        "redisPort" to (environment.config.propertyOrNull("redis.port")?.getString()?.toInt() ?: 6379)
    )

    // Create and start the email queue service
    val emailQueueService = EmailQueueFactory.createEmailQueue(queueType, emailService, queueConfig)
    emailQueueService.startProcessing(3) // Start with 3 workers

    // Add a shutdown hook to clean up resources
    monitor.subscribe(ApplicationStopped) {
        emailQueueService.shutdown()
    }

    routing {
        install(ContentNegotiation) {
            json(json = Json { ignoreUnknownKeys = true })
        }
        install(RequestValidation) {
            validate<EmailPayload> { emailPayload ->
                if (emailPayload.recipient.isEmpty() || emailPayload.subject.isEmpty() || emailPayload.body.isEmpty())
                    ValidationResult.Invalid("Recipient, subject, and body cannot be empty")
                else ValidationResult.Valid
            }
        }
        post("/api/email") {
            try {
                val emailPayload = call.receive<EmailPayload>()
                val queued = emailQueueService.queueEmail(emailPayload)

                if (queued) {
                    call.respond(HttpStatusCode.Accepted, mapOf("message" to "Email queued for delivery"))
                } else {
                    call.respond(
                        HttpStatusCode.InternalServerError,
                        mapOf("message" to "Failed to queue email")
                    )
                }
            } catch (e: Exception) {
                call.respond(
                    HttpStatusCode.BadRequest,
                    mapOf("message" to "Invalid email request: ${e.message}")
                )
            }
        }

        // Queue status endpoint
        get("/api/email/queue/status") {
            call.respond(emailQueueService.getMetrics())
        }

        // History endpoint
        get("/api/email/history") {
            val limit = call.request.queryParameters["limit"]?.toIntOrNull() ?: 100
            call.respond(emailQueueService.getProcessingHistory(limit))
        }
    }
}

@Serializable
data class EmailPayload(val recipient: String, val subject: String, val body: String)
data class EmailConfig(val from: String, val password: String, val host: String, val port: Int)

class KtorEmailService(emailConfig: EmailConfig) {
    private val logger = KtorSimpleLogger("KtorEmailService")
    private var host: String = "smtp.gmail.com"
    private var port: Int = 587
    private var emailFrom: String = "<EMAIL>"
    private var password: String = "<PASSWORD>"
    private var executionTime: Long = 0L
    private var props: Properties = Properties()

    // init email configurations
    init {
        // init email config
        emailConfig.let {
            host = it.host
            port = it.port
            emailFrom = it.from
            password = it.password
        }
        // init mail smtp props
        props = Properties().apply {
            put("mail.smtp.host", host)
            put("mail.smtp.port", port)
            put("mail.smtp.auth", true)
            put("mail.smtp.starttls.enable", false)
            put("mail.smtp.socketFactory.port", port)
            put("mail.smtp.socketFactory.class", "javax.net.ssl.SSLSocketFactory")
        }
        logger.info("KtorEmailService initialized")
    }

    /**
     * Sends an email using the SMTP protocol with the provided email payload details.
     *
     * @param emailPayload An instance of [EmailPayload] containing the recipient email address, subject, and body of the email.
     * @return A [Boolean] indicating the success or failure of the email-sending operation. Returns `true` if the email was sent successfully, `false` otherwise.
     */
    fun sendEmail(emailPayload: EmailPayload): Boolean {
        executionTime = System.currentTimeMillis()
        try {
            // build authenticator
            val authenticator = EmailAuthenticator(emailFrom, password)
            // build session
            val session = Session.getInstance(props, authenticator)
            // build a mime message
            val message: Message = MimeMessage(session).apply {
                setFrom(InternetAddress(emailFrom))
                setRecipient(Message.RecipientType.TO, InternetAddress(emailPayload.recipient))
                subject = subject
                setText(emailPayload.body)
            }
            // send a message
            Transport.send(message)
            logger.info("Email sent to $emailPayload.recipient")
            return true
        } catch (e: MessagingException) {
            logger.error("Failed to send email: $e")
            e.printStackTrace()
            return false
        } finally {
            executionTime = System.currentTimeMillis() - executionTime
            logger.info("Email execution time: $executionTime ms")
        }
    }

    /**
     * This class is responsible for providing email authentication for SMTP sessions.
     * It extends the Authenticator class and overrides the getPasswordAuthentication method
     * to supply the username and password required for authentication.
     *
     * @constructor Creates an instance of EmailAuthenticator with the specified email and password.
     * @param fromEmail The email address used to authenticate the SMTP session.
     * @param password The password associated with the email address for authentication.
     */
    class EmailAuthenticator(private val fromEmail: String, private val password: String) : Authenticator() {
        override fun getPasswordAuthentication(): PasswordAuthentication {
            return PasswordAuthentication(fromEmail, password)
        }
    }
}
/**
 * Interface for email queue services that abstract different queueing backends
 */
interface EmailQueueService {
    /**
     * Adds an email to the queue for asynchronous processing
     *
     * @param emailPayload The email to be queued
     * @param priority Optional priority level (higher means more priority)
     * @param delay Optional delay before the email should be processed
     * @return true if successfully queued, false otherwise
     */
    fun queueEmail(
        emailPayload: EmailPayload,
        priority: Int = 0,
        delay: Duration? = null
    ): Boolean

    /**
     * Starts processing the email queue
     *
     * @param workerCount Number of parallel workers to process emails
     */
    fun startProcessing(workerCount: Int = 1)

    /**
     * Stops processing the email queue
     */
    fun stopProcessing()

    /**
     * Gets the current size of the queue
     *
     * @return The number of emails in the queue, or -1 if size can't be determined
     */
    fun getQueueSize(): Int

    /**
     * Cleans up resources used by the queue service
     */
    fun shutdown()

    /**
     * Creates a scope for queue service-related coroutines
     *
     * @return A CoroutineScope for queue processing
     */
    fun getCoroutineScope(): CoroutineScope

    /**
     * Get queue status metrics
     *
     * @return A map of metric name to value
     */
    fun getMetrics(): EmailMetrics

    /**
     * Optional: Delete a specific email from the queue
     *
     * @param id The unique identifier of the email in the queue
     * @return true if successfully removed, false otherwise
     */
    fun removeFromQueue(id: String): Boolean {
        // Default implementation returns false
        return false
    }

    /**
     * Get email processing history
     *
     * @param limit Maximum number of history items to retrieve
     * @return List of email processing history items
     */
    fun getProcessingHistory(limit: Int = 100): List<EmailProcessingHistoryItem> {
        // Default implementation returns empty list
        return emptyList()
    }
}

enum class EmailQueueType {
    IN_MEMORY,
    REDIS,
    RABBIT_MQ,
    DATABASE
}

@Serializable
data class EmailProcessingHistoryItem(
    val id: String,
    val recipient: String,
    val subject: String,
    val queuedAt: Long,
    val processedAt: Long? = null,
    val status: EmailProcessingStatus,
    val attempts: Int,
    val errorMessage: String? = null
)

enum class EmailProcessingStatus {
    QUEUED,      // Initial state, waiting to be processed
    PROCESSING,  // Currently being processed
    SENT,        // Successfully sent
    FAILED,      // Failed to send after maximum retries
    DELETED      // Manually removed from queue
}

class InMemoryEmailQueueService(
    private val emailService: KtorEmailService
) : EmailQueueService {

    private val logger = KtorSimpleLogger("InMemoryEmailQueueService")
    private val coroutineScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val isProcessing = AtomicBoolean(false)

    // Email queue items with additional metadata
    private data class QueueItem(
        val id: String,
        val emailPayload: EmailPayload,
        val priority: Int,
        val queuedAt: Long,
        var processAfter: Long,
        var status: EmailProcessingStatus = EmailProcessingStatus.QUEUED,
        var attempts: Int = 0,
        var processedAt: Long? = null,
        var errorMessage: String? = null
    ) : Comparable<QueueItem> {
        override fun compareTo(other: QueueItem): Int {
            // First compare by processAfter time (delayed items)
            val timeComparison = processAfter.compareTo(other.processAfter)
            if (timeComparison != 0) return timeComparison

            // Then by priority (higher priority first)
            val priorityComparison = other.priority.compareTo(this.priority)
            if (priorityComparison != 0) return priorityComparison

            // Finally by queuedAt (FIFO for same priority)
            return queuedAt.compareTo(other.queuedAt)
        }
    }

    // Use a priority queue for ordering
    private val emailQueue = PriorityBlockingQueue<QueueItem>()

    // Keep history of processed emails
    private val emailHistory = ConcurrentHashMap<String, QueueItem>()

    // Metrics
    private val totalQueued = AtomicInteger(0)
    private val totalProcessed = AtomicInteger(0)
    private val totalFailed = AtomicInteger(0)

    override fun queueEmail(
        emailPayload: EmailPayload,
        priority: Int,
        delay: Duration?
    ): Boolean {
        return try {
            val now = System.currentTimeMillis()
            val processAfter = if (delay != null) now + delay.toMillis() else now

            val id = UUID.randomUUID().toString()
            val queueItem = QueueItem(
                id = id,
                emailPayload = emailPayload,
                priority = priority,
                queuedAt = now,
                processAfter = processAfter
            )

            emailQueue.offer(queueItem)
            emailHistory[id] = queueItem
            totalQueued.incrementAndGet()

            logger.info("Email to ${emailPayload.recipient} queued with ID: $id")
            true
        } catch (e: Exception) {
            logger.error("Failed to queue email: ${e.message}")
            false
        }
    }

    override fun startProcessing(workerCount: Int) {
        if (isProcessing.getAndSet(true)) {
            return
        }

        repeat(workerCount) { workerId ->
            coroutineScope.launch {
                logger.info("Starting worker #$workerId")
                while (isActive && isProcessing.get()) {
                    processNextEmail(workerId)
                    delay(100) // Small delay to prevent CPU hogging
                }
            }
        }
    }

    private suspend fun processNextEmail(workerId: Int) {
        try {
            val now = System.currentTimeMillis()

            // Peek at the next item to see if it's time to process
            val nextItem = emailQueue.peek() ?: return

            // Skip if the item is scheduled for future processing
            if (nextItem.processAfter > now) return

            // Remove the item from the queue
            emailQueue.poll() ?: return

            // Update status
            nextItem.status = EmailProcessingStatus.PROCESSING

            withContext(Dispatchers.IO) {
                logger.info("Worker #$workerId processing email to ${nextItem.emailPayload.recipient}")
                nextItem.attempts++

                try {
                    val success = emailService.sendEmail(nextItem.emailPayload)

                    if (success) {
                        nextItem.status = EmailProcessingStatus.SENT
                        nextItem.processedAt = System.currentTimeMillis()
                        totalProcessed.incrementAndGet()
                        logger.info("Email ${nextItem.id} sent successfully")
                    } else {
                        handleFailedEmail(nextItem)
                    }
                } catch (e: Exception) {
                    nextItem.errorMessage = e.message
                    handleFailedEmail(nextItem)
                }
            }
        } catch (e: Exception) {
            logger.error("Error processing email from queue: ${e.message}")
        }
    }

    private fun handleFailedEmail(item: QueueItem) {
        // Maximum retry attempts
        if (item.attempts >= 3) {
            item.status = EmailProcessingStatus.FAILED
            item.processedAt = System.currentTimeMillis()
            totalFailed.incrementAndGet()
            logger.error("Email ${item.id} failed after ${item.attempts} attempts")
        } else {
            // Exponential backoff for retries (1s, 4s, 9s)
            val backoffMillis = (item.attempts * item.attempts) * 1000L
            item.processAfter = System.currentTimeMillis() + backoffMillis
            item.status = EmailProcessingStatus.QUEUED

            emailQueue.offer(item)
            logger.warn("Email ${item.id} failed, will retry in ${backoffMillis/1000} seconds (attempt ${item.attempts})")
        }
    }

    override fun stopProcessing() {
        isProcessing.set(false)
        logger.info("Email queue processor stopped")
    }

    override fun getQueueSize(): Int = emailQueue.size

    override fun shutdown() {
        stopProcessing()
        coroutineScope.cancel()
    }

    override fun getCoroutineScope(): CoroutineScope = coroutineScope

    override fun getMetrics(): EmailMetrics {
        return EmailMetrics(
            queueSize = emailQueue.size.toLong(),
            totalQueued = totalQueued.get().toLong(),
            totalProcessed = totalProcessed.get().toLong(),
            totalFailed = totalFailed.get().toLong(),
            processingActive = isProcessing.get()
        )
    }

    override fun removeFromQueue(id: String): Boolean {
        val item = emailHistory[id] ?: return false

        if (item.status == EmailProcessingStatus.QUEUED) {
            // Remove from active queue
            val removed = emailQueue.removeIf { it.id == id }
            if (removed) {
                item.status = EmailProcessingStatus.DELETED
                return true
            }
        }
        return false
    }

    override fun getProcessingHistory(limit: Int): List<EmailProcessingHistoryItem> {
        return emailHistory.values
            .sortedByDescending { it.queuedAt }
            .take(limit)
            .map {
                EmailProcessingHistoryItem(
                    id = it.id,
                    recipient = it.emailPayload.recipient,
                    subject = it.emailPayload.subject,
                    queuedAt = it.queuedAt,
                    processedAt = it.processedAt,
                    status = it.status,
                    attempts = it.attempts,
                    errorMessage = it.errorMessage
                )
            }
    }
}

class RedisEmailQueueService(
    private val emailService: KtorEmailService,
    private val redisHost: String = "localhost",
    private val redisPort: Int = 6379
) : EmailQueueService {

    private val logger = KtorSimpleLogger("RedisEmailQueueService")
    private val json = Json { ignoreUnknownKeys = true }
    private val coroutineScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val isProcessing = AtomicBoolean(false)

    // Redis keys
    private val queueKey = "email:queue"
    private val processingKey = "email:processing"
    private val historyKey = "email:history"
    private val metricsKey = "email:metrics"

    // Jedis connection pool
    private val jedisPool: JedisPool

    // For local metrics
    private val totalProcessed = AtomicInteger(0)
    private val totalFailed = AtomicInteger(0)

    @Serializable
    private data class RedisQueueItem(
        val id: String,
        val emailPayload: EmailPayload,
        val priority: Int,
        val queuedAt: Long,
        var status: String,
        var attempts: Int = 0,
        var processedAt: Long? = null,
        var errorMessage: String? = null
    )

    init {
        val poolConfig = JedisPoolConfig().apply {
            maxTotal = 10
            maxIdle = 5
            minIdle = 1
            testOnBorrow = true
            testOnReturn = true
            testWhileIdle = true
            blockWhenExhausted = true
        }

        jedisPool = JedisPool(poolConfig, redisHost, redisPort)
        logger.info("RedisEmailQueueService initialized with Redis at $redisHost:$redisPort")

        // Initialize metrics in Redis
        jedisPool.resource.use { jedis ->
            if (!jedis.exists(metricsKey)) {
                jedis.hset(metricsKey, mapOf(
                    "totalQueued" to "0",
                    "totalProcessed" to "0",
                    "totalFailed" to "0"
                ))
            }
        }
    }

    override fun queueEmail(
        emailPayload: EmailPayload,
        priority: Int,
        delay: Duration?
    ): Boolean {
        return try {
            jedisPool.resource.use { jedis ->
                val now = System.currentTimeMillis()
                val id = UUID.randomUUID().toString()

                val queueItem = RedisQueueItem(
                    id = id,
                    emailPayload = emailPayload,
                    priority = priority,
                    queuedAt = now,
                    status = "QUEUED"
                )

                val itemJson = json.encodeToString(queueItem)

                // Store the item data
                jedis.hset(historyKey, id, itemJson)

                // Add to the sorted set queue with score based on priority and time
                val processTime = if (delay != null) now + delay.toMillis() else now
                val score = processTime - (priority * 10000) // Higher priority = lower score = processed earlier

                jedis.zadd(queueKey, score.toDouble(), id)

                // Update metrics
                jedis.hincrBy(metricsKey, "totalQueued", 1)

                logger.info("Email to ${emailPayload.recipient} queued with ID: $id")
                true
            }
        } catch (e: Exception) {
            logger.error("Failed to queue email: ${e.message}")
            false
        }
    }

    override fun startProcessing(workerCount: Int) {
        if (isProcessing.getAndSet(true)) {
            return
        }

        repeat(workerCount) { workerId ->
            coroutineScope.launch {
                logger.info("Starting worker #$workerId")
                while (isActive && isProcessing.get()) {
                    processNextEmail(workerId)
                    delay(100) // Small delay to prevent CPU hogging
                }
            }
        }
    }

    private suspend fun processNextEmail(workerId: Int) {
        var id: String? = null

        try {
            jedisPool.resource.use { jedis ->
                val now = System.currentTimeMillis()

                // Get items with score <= current time (ready to be processed)
                val nextItems = jedis.zrangeByScore(queueKey, 0.0, now.toDouble(), 0, 1)
                if (nextItems.isEmpty()) return

                id = nextItems.first()

                // Move from queue to processing set
                if (jedis.zrem(queueKey, id) == 0L) {
                    // Item was already taken by another worker
                    return
                }

                // Get the item data
                val itemJson = jedis.hget(historyKey, id) ?: return
                val item = json.decodeFromString<RedisQueueItem>(itemJson)

                // Update status
                item.status = "PROCESSING"
                item.attempts++
                jedis.hset(historyKey, id, json.encodeToString(item))

                // Process the email outside the Redis transaction
                withContext(Dispatchers.IO) {
                    logger.info("Worker #$workerId processing email to ${item.emailPayload.recipient}")

                    try {
                        val success = emailService.sendEmail(item.emailPayload)

                        jedisPool.resource.use { updateJedis ->
                            if (success) {
                                item.status = "SENT"
                                item.processedAt = System.currentTimeMillis()
                                updateJedis.hset(historyKey, id, json.encodeToString(item))
                                updateJedis.hincrBy(metricsKey, "totalProcessed", 1)
                                totalProcessed.incrementAndGet()
                                logger.info("Email $id sent successfully")
                            } else {
                                handleFailedEmail(updateJedis, item)
                            }
                        }
                    } catch (e: Exception) {
                        jedisPool.resource.use { updateJedis ->
                            item.errorMessage = e.message
                            handleFailedEmail(updateJedis, item)
                        }
                    }
                }
            }
        } catch (e: Exception) {
            logger.error("Error processing email from queue: ${e.message}")

            // If we had an ID but failed to process it, move it back to the queue
            if (id != null) {
                try {
                    jedisPool.resource.use { jedis ->
                        val itemJson = jedis.hget(historyKey, id) ?: return
                        val item = json.decodeFromString<RedisQueueItem>(itemJson)

                        // Re-queue with exponential backoff
                        val backoffMillis = (item.attempts * item.attempts) * 1000L
                        val processTime = System.currentTimeMillis() + backoffMillis
                        val score = processTime - (item.priority * 10000)

                        jedis.zadd(queueKey, score.toDouble(), id)
                    }
                } catch (retryEx: Exception) {
                    logger.error("Failed to re-queue item $id: ${retryEx.message}")
                }
            }
        }
    }

    private fun handleFailedEmail(jedis: Jedis, item: RedisQueueItem) {
        if (item.attempts >= 3) {
            item.status = "FAILED"
            item.processedAt = System.currentTimeMillis()
            jedis.hset(historyKey, item.id, json.encodeToString(item))
            jedis.hincrBy(metricsKey, "totalFailed", 1)
            totalFailed.incrementAndGet()
            logger.error("Email ${item.id} failed after ${item.attempts} attempts: ${item.errorMessage}")
        } else {
            // Exponential backoff
            val backoffMillis = (item.attempts * item.attempts) * 1000L
            val processTime = System.currentTimeMillis() + backoffMillis
            val score = processTime - (item.priority * 10000)

            item.status = "QUEUED"
            jedis.hset(historyKey, item.id, json.encodeToString(item))
            jedis.zadd(queueKey, score.toDouble(), item.id)

            logger.warn("Email ${item.id} failed, will retry in ${backoffMillis/1000} seconds (attempt ${item.attempts})")
        }
    }

    override fun stopProcessing() {
        isProcessing.set(false)
        logger.info("Email queue processor stopped")
    }

    override fun getQueueSize(): Int {
        return try {
            jedisPool.resource.use { jedis ->
                jedis.zcard(queueKey).toInt()
            }
        } catch (e: Exception) {
            logger.error("Failed to get queue size: ${e.message}")
            -1
        }
    }

    override fun shutdown() {
        stopProcessing()
        coroutineScope.cancel()
        jedisPool.close()
    }

    override fun getCoroutineScope(): CoroutineScope = coroutineScope

    override fun getMetrics(): EmailMetrics {
        try {
            jedisPool.resource.use { jedis ->
                val redisMetrics = jedis.hgetAll(metricsKey)
                return EmailMetrics(
                    queueSize = jedis.zcard(queueKey),
                    totalQueued = (redisMetrics["totalQueued"]?.toLongOrNull() ?: 0),
                    totalProcessed = (redisMetrics["totalProcessed"]?.toLongOrNull() ?: 0),
                    totalFailed = (redisMetrics["totalFailed"]?.toLongOrNull() ?: 0),
                    processingActive = isProcessing.get()
                )
            }
        } catch (e: Exception) {
            logger.error("Failed to get metrics: ${e.message}")
            return EmailMetrics(
                error = "Failed to retrieve metrics",
                processingActive = isProcessing.get()
            )
        }
    }

    override fun removeFromQueue(id: String): Boolean {
        return try {
            jedisPool.resource.use { jedis ->
                val removed = jedis.zrem(queueKey, id)
                if (removed > 0) {
                    val itemJson = jedis.hget(historyKey, id) ?: return false
                    val item = json.decodeFromString<RedisQueueItem>(itemJson)
                    item.status = "DELETED"
                    jedis.hset(historyKey, id, json.encodeToString(item))
                    true
                } else {
                    false
                }
            }
        } catch (e: Exception) {
            logger.error("Failed to remove item $id: ${e.message}")
            false
        }
    }

    override fun getProcessingHistory(limit: Int): List<EmailProcessingHistoryItem> {
        return try {
            jedisPool.resource.use { jedis ->
                // Get all history keys (could be optimized with Redis sets)
                val historyItems = jedis.hgetAll(historyKey)

                historyItems.values
                    .map { json.decodeFromString<RedisQueueItem>(it) }
                    .sortedByDescending { it.queuedAt }
                    .take(limit)
                    .map { item ->
                        EmailProcessingHistoryItem(
                            id = item.id,
                            recipient = item.emailPayload.recipient,
                            subject = item.emailPayload.subject,
                            queuedAt = item.queuedAt,
                            processedAt = item.processedAt,
                            status = when(item.status) {
                                "QUEUED" -> EmailProcessingStatus.QUEUED
                                "PROCESSING" -> EmailProcessingStatus.PROCESSING
                                "SENT" -> EmailProcessingStatus.SENT
                                "FAILED" -> EmailProcessingStatus.FAILED
                                "DELETED" -> EmailProcessingStatus.DELETED
                                else -> EmailProcessingStatus.QUEUED
                            },
                            attempts = item.attempts,
                            errorMessage = item.errorMessage
                        )
                    }
            }
        } catch (e: Exception) {
            logger.error("Failed to get processing history: ${e.message}")
            emptyList()
        }
    }
}

@Serializable
data class EmailMetrics(
    val queueSize: Long = 0,
    val totalQueued: Long = 0,
    val totalProcessed: Long = 0,
    val totalFailed: Long = 0,
    val error: String = "",
    val processingActive: Boolean = false
)

class EmailQueueFactory {
    companion object {
        fun createEmailQueue(
            type: EmailQueueType,
            emailService: KtorEmailService,
            config: Map<String, Any> = emptyMap()
        ): EmailQueueService {
            return when (type) {
                EmailQueueType.IN_MEMORY -> InMemoryEmailQueueService(emailService)
                EmailQueueType.REDIS -> RedisEmailQueueService(
                    emailService,
                    redisHost = config["redisHost"] as? String ?: "localhost",
                    redisPort = config["redisPort"] as? Int ?: 6379
                )
                EmailQueueType.RABBIT_MQ -> {
                    // RabbitMQEmailService implementation
                    throw NotImplementedError("RabbitMQ implementation not available yet")
                }
                EmailQueueType.DATABASE -> {
                    // DatabaseEmailQueueService implementation
                    throw NotImplementedError("Database implementation not available yet")
                }
            }
        }
    }
}

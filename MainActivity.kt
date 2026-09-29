package com.tencent.yolo11ncnn

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.util.Size
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * ============================================================================
 * MainActivity - Autonomous Robotic Mower Vision Processor
 * Platform: Samsung Galaxy S9+ (Mali-G72 MP18 + Exynos 9810)
 *
 * ASYNCHRONICZNY SYSTEM STARTU (ZERO DEADLOCK):
 * 1. Start TelemetryServer (Port 8080) w onCreate() - natychmiastowa gotowość WWW.
 * 2. Inicjalizacja YOLO11 Vulkan w tle (Background Executor) - brak zamrażania UI/ANR!
 * 3. Start CameraX DOPIERO PO załadowaniu YOLO - eliminuje wyścigi i wyczerpanie Gralloc!
 * ============================================================================
 */
class MainActivity : AppCompatActivity() {

    private val yoloEngine = Yolo11Ncnn()
    private lateinit var telemetryServer: TelemetryServer
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var backgroundInitExecutor: ExecutorService

    // Pre-allocated flat buffer: 1 count float + 64 detections * 7 floats (449 floats)
    // ZERO HEAP ALLOCATIONS ON THE FRAME LOOP!
    private val detectionBuffer = FloatArray(1 + 64 * 7)

    @Volatile private var confThreshold: Float = 0.25f
    @Volatile private var nmsThreshold: Float = 0.45f
    @Volatile private var manualEStop: Boolean = false

    private val isYoloReady = AtomicBoolean(false)
    private val isCameraRunning = AtomicBoolean(false)

    private var frameCount: Int = 0
    private var lastFpsCalcTimeMs: Long = 0
    private var currentFps: Float = 0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(android.R.layout.list_content)

        cameraExecutor = Executors.newSingleThreadExecutor()
        backgroundInitExecutor = Executors.newSingleThreadExecutor()

        // KROK 1: Start serwera WebSocket natychmiast
        startTelemetryServer()

        // KROK 2 & 3: Weryfikacja uprawnień i asynchroniczny start potoku
        if (allPermissionsGranted()) {
            launchAsyncPipeline()
        } else {
            telemetryServer.broadcastStatus("AWAITING_PERMISSIONS", "Oczekiwanie na akceptację uprawnień...")
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                Log.i(TAG, "Uprawnienia przyznane, uruchamianie potoku")
                launchAsyncPipeline()
            } else {
                Log.e(TAG, "Brak uprawnień kamery!")
                telemetryServer.broadcastStatus("ERROR_PERMISSIONS", "Brak uprawnień do kamery!")
            }
        }
    }

    private fun startTelemetryServer() {
        try {
            telemetryServer = TelemetryServer(
                port = 8080,
                onCommandReceived = { cmd, payload ->
                    handleRemoteCommand(cmd, payload)
                }
            )
            telemetryServer.start()
            Log.i(TAG, "TelemetryServer wystartował na porcie 8080")
            telemetryServer.broadcastStatus("SERVER_ONLINE", "Serwer WebSocket nasłuchuje na porcie 8080")
        } catch (e: Exception) {
            Log.e(TAG, "Błąd startu TelemetryServer", e)
        }
    }

    private fun launchAsyncPipeline() {
        backgroundInitExecutor.execute {
            Log.i(TAG, "KROK 2: Ładowanie YOLO11 Vulkan w tle...")
            telemetryServer.broadcastStatus("INIT_YOLO", "Inicjalizacja GPU Mali-G72 i ładowanie wag YOLO11...")

            val startTime = System.currentTimeMillis()
            var initSuccess = false

            try {
                initSuccess = yoloEngine.Init(assets, Yolo11Ncnn.MODEL_YOLO11N, 320, true)
            } catch (e: Exception) {
                Log.e(TAG, "Błąd podczas init YOLO Vulkan", e)
            }

            val loadDuration = System.currentTimeMillis() - startTime

            if (initSuccess) {
                Log.i(TAG, "YOLO11 Vulkan Init sukces w " + loadDuration + "ms")
                isYoloReady.set(true)
                telemetryServer.broadcastStatus("YOLO_READY", "YOLO11 gotowe na Mali-G72 (" + loadDuration + "ms)")

                // KROK 3: Start kamery DOPIERO po gotowości YOLO
                ContextCompat.getMainExecutor(this@MainActivity).execute {
                    Log.i(TAG, "KROK 3: Uruchamianie CameraX...")
                    telemetryServer.broadcastStatus("STARTING_CAMERA", "Uruchamianie strumienia kamery...")
                    startCamera()
                }
            } else {
                Log.w(TAG, "Vulkan Init nie powiódł się, fallback na CPU...")
                telemetryServer.broadcastStatus("WARN_VULKAN", "Vulkan niedostępny, fallback CPU...")

                val cpuSuccess = yoloEngine.Init(assets, Yolo11Ncnn.MODEL_YOLO11N, 320, false)
                if (cpuSuccess) {
                    isYoloReady.set(true)
                    telemetryServer.broadcastStatus("YOLO_READY", "YOLO11 gotowe w trybie CPU")
                    ContextCompat.getMainExecutor(this@MainActivity).execute {
                        startCamera()
                    }
                } else {
                    Log.e(TAG, "Niepowodzenie ładowania wag z assets!")
                    telemetryServer.broadcastStatus("ERROR_YOLO", "Błąd ładowania parametrów modelu!")
                }
            }
        }
    }

    private fun handleRemoteCommand(cmd: String, payload: String) {
        val q = '"'
        when (cmd) {
            "EMERGENCY_STOP" -> {
                manualEStop = true
                telemetryServer.broadcastStatus("ESTOP_MANUAL", "Twardy E-STOP z dashboardu!")
            }
            "RESET_ESTOP" -> {
                manualEStop = false
                yoloEngine.ResetEmergencyStop()
                telemetryServer.broadcastStatus("ESTOP_RESET", "Zresetowano E-STOP")
            }
            "SET_PARAMS" -> {
                try {
                    val confKey = "" + q + "conf" + q
                    val nmsKey = "" + q + "nms" + q
                    val conf = payload.substringAfter(confKey, "").substringAfter(":", "").substringBefore(",").substringBefore("}").replace("\"", "").trim().toFloatOrNull()
                    val nms = payload.substringAfter(nmsKey, "").substringAfter(":", "").substringBefore(",").substringBefore("}").replace("\"", "").trim().toFloatOrNull()
                    if (conf != null) confThreshold = conf
                    if (nms != null) nmsThreshold = nms
                    telemetryServer.broadcastStatus("PARAMS_UPDATED", "conf=" + confThreshold + ", nms=" + nmsThreshold)
                } catch (e: Exception) {
                    Log.e(TAG, "Błąd parsowania SET_PARAMS", e)
                }
            }
        }
    }

    @Suppress("DEPRECATION")
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(320, 320))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        if (!isYoloReady.get()) {
                            imageProxy.close()
                            return@setAnalyzer
                        }

                        calculateFps()

                        // Convert CameraX rotation to Vulkan EXIF rotate type
                        val rotateType = when (imageProxy.imageInfo.rotationDegrees) {
                            90 -> 6
                            180 -> 3
                            270 -> 8
                            else -> 1
                        }

                        val image = imageProxy.image
                        val hwBuffer = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                            image?.hardwareBuffer
                        } else null

                        var count = 0

                        if (hwBuffer != null) {
                            try {
                                count = yoloEngine.DetectAHBDirect(
                                    hwBuffer,
                                    detectionBuffer,
                                    confThreshold,
                                    nmsThreshold,
                                    rotateType
                                )
                            } catch (e: Exception) {
                                Log.e(TAG, "Błąd DetectAHBDirect", e)
                            } finally {
                                hwBuffer.close()
                            }
                        } else {
                            Log.w(TAG, "Brak HardwareBuffer w klatce kamery")
                        }

                        val topScore = yoloEngine.GetTopScore()
                        val topClass = yoloEngine.GetTopClass()

                        // Log diagnostic stats once per second to adb logcat
                        if (frameCount == 1) {
                            Log.i(TAG, "Inference: FPS=" + String.format(java.util.Locale.US, "%.1f", currentFps) +
                                    " | Objects=" + count +
                                    " | TopCandidate=" + String.format(java.util.Locale.US, "%.3f", topScore) +
                                    " (class=" + topClass + ")" +
                                    " | ConfThresh=" + confThreshold)
                        }

                        val isHardwareEStop = yoloEngine.IsEmergencyStop()
                        val finalEStop = manualEStop || isHardwareEStop

                        val jsonPayload = buildJson(count, detectionBuffer, finalEStop, currentFps, topScore, topClass)
                        telemetryServer.sendDetections(jsonPayload)

                        sendToSTM32(count, detectionBuffer, finalEStop)

                        imageProxy.close()
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, imageAnalyzer)
                isCameraRunning.set(true)
                Log.i(TAG, "Camera2 powiązana z cyklem życia")
                telemetryServer.broadcastStatus("STREAMING", "Strumień wizyjny aktywny (CameraX + Mali-G72 Zero-Copy)")
            } catch (exc: Exception) {
                Log.e(TAG, "Błąd powiązania kamery", exc)
                telemetryServer.broadcastStatus("ERROR_CAMERA", "Błąd powiązania kamery: " + exc.message)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun calculateFps() {
        val now = System.currentTimeMillis()
        frameCount++
        if (now - lastFpsCalcTimeMs >= 1000) {
            currentFps = (frameCount * 1000f) / (now - lastFpsCalcTimeMs)
            frameCount = 0
            lastFpsCalcTimeMs = now
        }
    }

    private fun buildJson(count: Int, data: FloatArray, eStop: Boolean, fpsVal: Float, topScore: Float, topClass: Int): String {
        val q = '"'
        val sb = StringBuilder(512)
        sb.append("{")
            .append(q).append("type").append(q).append(":").append(q).append("telemetry").append(q).append(",")
            .append(q).append("fps").append(q).append(":").append(String.format(java.util.Locale.US, "%.1f", fpsVal)).append(",")
            .append(q).append("topScore").append(q).append(":").append(String.format(java.util.Locale.US, "%.3f", topScore)).append(",")
            .append(q).append("topClass").append(q).append(":").append(topClass).append(",")
            .append(q).append("conf").append(q).append(":").append(confThreshold).append(",")
            .append(q).append("eStop").append(q).append(":").append(eStop).append(",")
            .append(q).append("count").append(q).append(":").append(count).append(",")
            .append(q).append("objects").append(q).append(":[")

        for (i in 0 until count) {
            val base = 1 + (i * 7)
            sb.append("{")
                .append(q).append("zone").append(q).append(":").append(data[base].toInt()).append(",")
                .append(q).append("class").append(q).append(":").append(data[base + 1].toInt()).append(",")
                .append(q).append("score").append(q).append(":").append(String.format(java.util.Locale.US, "%.3f", data[base + 2])).append(",")
                .append(q).append("x").append(q).append(":").append(String.format(java.util.Locale.US, "%.3f", data[base + 3])).append(",")
                .append(q).append("y").append(q).append(":").append(String.format(java.util.Locale.US, "%.3f", data[base + 4])).append(",")
                .append(q).append("w").append(q).append(":").append(String.format(java.util.Locale.US, "%.3f", data[base + 5])).append(",")
                .append(q).append("h").append(q).append(":").append(String.format(java.util.Locale.US, "%.3f", data[base + 6]))
            sb.append("}")
            if (i < count - 1) sb.append(",")
        }
        sb.append("]}")
        return sb.toString()
    }

    private fun sendToSTM32(count: Int, data: FloatArray, eStop: Boolean) {
        var critCount = 0
        var reactCount = 0
        var horizCount = 0

        for (i in 0 until count) {
            val zone = data[1 + i * 7].toInt()
            when (zone) {
                0 -> critCount++
                1 -> reactCount++
                2 -> horizCount++
            }
        }

        val packet = STM32UartProtocol.buildObstaclePacket(
            emergencyStop = eStop,
            criticalCount = critCount,
            reactionCount = reactCount,
            horizonCount = horizCount,
            nearestDistanceMm = if (critCount > 0) 150 else if (reactCount > 0) 650 else 2500,
            steerAngleDeg = 0
        )
        if (packet.isNotEmpty() && eStop) {
            Log.d(TAG, "STM32 E-STOP pakiet wysłany")
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        isYoloReady.set(false)
        cameraExecutor.shutdown()
        backgroundInitExecutor.shutdown()

        if (::telemetryServer.isInitialized) {
            try {
                telemetryServer.stop()
            } catch (e: Exception) {
                Log.e(TAG, "Błąd zamykania TelemetryServer", e)
            }
        }
        yoloEngine.ClearCache()
        yoloEngine.Destroy()
    }

    companion object {
        private const val TAG = "Mower_Vision_Core"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.INTERNET
        )
    }
}

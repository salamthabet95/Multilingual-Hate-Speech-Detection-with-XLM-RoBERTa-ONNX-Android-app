package com.hatespeechdetector

import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import okhttp3.OkHttpClient
import java.util.concurrent.TimeUnit
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST

data class PredictionRequest(val text: String)
data class PredictionResponse(val prediction: Int, val confidence: Float, val model_used: String)

interface HateSpeechAPI {
    @POST("predict")
    fun predict(@Body request: PredictionRequest): Call<PredictionResponse>
    
    @GET("health")
    fun healthCheck(): Call<HealthResponse>
}

data class HealthResponse(
    val status: String,
    val transformer_loaded: Boolean
)

class MainActivity : AppCompatActivity() {
    
    private lateinit var api: HateSpeechAPI
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Initialize Retrofit with timeout configuration
        val baseUrl = getString(R.string.api_base_url)
        val timeout = resources.getInteger(R.integer.api_timeout_seconds).toLong()
        val okHttpClient = OkHttpClient.Builder()
            .connectTimeout(timeout, TimeUnit.SECONDS)
            .readTimeout(timeout, TimeUnit.SECONDS)
            .writeTimeout(timeout, TimeUnit.SECONDS)
            .build()
        
        val retrofit = Retrofit.Builder()
            .baseUrl(baseUrl)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
        
        api = retrofit.create(HateSpeechAPI::class.java)
        
        // Get UI elements
        val textInput = findViewById<EditText>(R.id.textInput)
        val predictButton = findViewById<Button>(R.id.predictButton)
        val resultText = findViewById<TextView>(R.id.resultText)
        val confidenceText = findViewById<TextView>(R.id.confidenceText)
        val modelUsedText = findViewById<TextView>(R.id.modelUsedText)
        val statusText = findViewById<TextView>(R.id.statusText)
        
        // Setup predict button
        predictButton.setOnClickListener {
            val text = textInput.text.toString().trim()
            if (text.isEmpty()) {
                Toast.makeText(this, "Please enter some text", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            
            predictHateSpeech(text, resultText, confidenceText, modelUsedText, statusText)
        }
        
        // Test API connection on startup
        testApiConnection(statusText)
    }
    
    private fun testApiConnection(statusText: TextView) {
        statusText.text = "Connecting..."
        statusText.setTextColor(ContextCompat.getColor(this, R.color.status_connecting))
        
        // Use proper health check endpoint
        api.healthCheck().enqueue(object : Callback<HealthResponse> {
            override fun onResponse(call: Call<HealthResponse>, response: Response<HealthResponse>) {
                if (response.isSuccessful) {
                    val health = response.body()
                    if (health != null) {
                        val status = if (health.status == "healthy") {
                            "Connected ✓"
                        } else {
                            "Service Unavailable"
                        }
                        statusText.text = status
                        statusText.setTextColor(
                            if (health.status == "healthy") 
                                ContextCompat.getColor(this@MainActivity, R.color.status_connected)
                            else 
                                ContextCompat.getColor(this@MainActivity, R.color.status_error)
                        )
                    } else {
                        statusText.text = "No Response"
                        statusText.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.status_error))
                    }
                } else {
                    statusText.text = "Connection Error"
                    statusText.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.status_error))
                }
            }
            
            override fun onFailure(call: Call<HealthResponse>, t: Throwable) {
                statusText.text = "Connection Failed"
                statusText.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.status_error))
            }
        })
    }
    
    private fun predictHateSpeech(
        text: String,
        resultText: TextView,
        confidenceText: TextView,
        modelUsedText: TextView,
        statusText: TextView
    ) {
        statusText.text = "Analyzing..."
        statusText.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.status_connecting))
        
        val request = PredictionRequest(text)
        api.predict(request).enqueue(object : Callback<PredictionResponse> {
            override fun onResponse(call: Call<PredictionResponse>, response: Response<PredictionResponse>) {
                if (response.isSuccessful) {
                    val result = response.body()
                    if (result != null) {
                        // Update UI with results
                        val predictionText = if (result.prediction == 1) "🚨 Hate Speech" else "✅ Safe Content"
                        val predictionColor = if (result.prediction == 1) 
                            ContextCompat.getColor(this@MainActivity, R.color.hate_color) 
                        else 
                            ContextCompat.getColor(this@MainActivity, R.color.safe_color)
                        
                        resultText.text = predictionText
                        resultText.setTextColor(predictionColor)
                        
                        confidenceText.text = "${(result.confidence * 100).toInt()}%"
                        
                        statusText.text = "Analysis Complete ✓"
                        statusText.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.status_connected))
                    } else {
                        showError("No response data", statusText)
                    }
                } else {
                    showError("API Error: ${response.code()}", statusText)
                }
            }
            
            override fun onFailure(call: Call<PredictionResponse>, t: Throwable) {
                showError("Network Error: ${t.message}", statusText)
            }
        })
    }
    
    private fun showError(message: String, statusText: TextView) {
        statusText.text = message
        statusText.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.status_error))
    }
}

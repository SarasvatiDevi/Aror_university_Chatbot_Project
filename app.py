from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import time

# Import your production bot
from rag_bot import ProductionRAGBot

app = FastAPI(
    title="Aror University AI Chatbot",
    description="Advanced AI Assistant for Aror University - Production Ready",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(current_dir, "static")
data_file = os.path.join(current_dir, "data", "university_data.json")

print(f"📁 Current directory: {current_dir}")
print(f"📁 Frontend directory: {frontend_dir}")
print(f"📁 Data file: {data_file}")

# Check if required directories exist
if not os.path.exists(frontend_dir):
    print(f"❌ WARNING: Frontend directory not found: {frontend_dir}")
if not os.path.exists(os.path.dirname(data_file)):
    print(f"❌ WARNING: Data directory not found: {os.path.dirname(data_file)}")

# Initialize production chatbot
print("🚀 Initializing Production RAG Bot...")
start_time = time.time()
service_start_time = time.time()

try:
    chatbot = ProductionRAGBot(data_file)
    initialization_time = time.time() - start_time
    print(f"✅ Production RAG Bot initialized in {initialization_time:.2f} seconds!")

    # Ensure the chatbot has required attributes
    if not hasattr(chatbot, 'ai_enabled'):
        chatbot.ai_enabled = True
    if not hasattr(chatbot, 'qa_pairs'):
        chatbot.qa_pairs = []

except Exception as e:
    print(f"❌ Error initializing chatbot: {e}")


    # Fallback bot with all required attributes
    class FallbackBot:
        def __init__(self):
            self.qa_pairs = []
            self.ai_enabled = False

        def get_response(self, query):
            return "🤖 Please contact us directly: 📱 0325-2278377 | 📧 admissions@aror.edu.pk"


    chatbot = FallbackBot()


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    success: bool
    response_time: float


# Serve static files only if static directory exists
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    print("✅ Static files mounted at /static")
else:
    print("❌ Frontend directory not found - static files not available")


@app.get("/")
async def serve_home():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return JSONResponse(
            status_code=200,
            content={
                "message": "Aror University Chatbot API is running!",
                "frontend_status": "not_found",
                "instructions": "Frontend files not found. Use API endpoints directly.",
                "endpoints": {
                    "health": "/health",
                    "ask_question": "/api/ask (POST)",
                    "stats": "/api/stats"
                }
            }
        )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Aror University Chatbot API",
        "version": "1.0.0",
        "timestamp": time.time(),
        "data_loaded": len(chatbot.qa_pairs) if hasattr(chatbot, 'qa_pairs') else 0,
        "ai_enabled": chatbot.ai_enabled if hasattr(chatbot, 'ai_enabled') else False,
        "frontend_available": os.path.exists(frontend_dir)
    }


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    start_time = time.time()

    try:
        if not request.question or not request.question.strip():
            return AnswerResponse(
                answer="Please ask a question about Aror University.",
                success=False,
                response_time=time.time() - start_time
            )

        response_dict = chatbot.get_response(request.question)
        # Extract just the answer string from the dictionary
        answer = response_dict['answer'] if isinstance(response_dict, dict) else response_dict

        response_time = time.time() - start_time

        return AnswerResponse(
            answer=answer,
            success=True,
            response_time=response_time
        )

    except Exception as e:
        print(f"API Error: {e}")
        return AnswerResponse(
            answer="Sorry, I encountered an error. Please contact 📱 0325-2278377 for assistance.",
            success=False,
            response_time=time.time() - start_time
        )


@app.get("/api/stats")
async def get_stats():
    return {
        "total_questions": len(chatbot.qa_pairs) if hasattr(chatbot, 'qa_pairs') else 0,
        "ai_enabled": chatbot.ai_enabled if hasattr(chatbot, 'ai_enabled') else False,
        "service_uptime": time.time() - service_start_time
    }


# Debug endpoint to see all routes
@app.get("/routes")
async def get_routes():
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "name": route.name,
            "methods": getattr(route, "methods", None)
        })
    return routes

@app.post("/api/clear_cache")
async def clear_cache():
    """Clear the bot's cache to improve performance"""
    try:
        if hasattr(chatbot, 'clear_cache'):
            chatbot.clear_cache()
            return {"message": "Cache cleared successfully", "success": True}
        else:
            return {"message": "Cache clearing not supported", "success": False}
    except Exception as e:
        return {"message": f"Error clearing cache: {str(e)}", "success": False}
if __name__ == "__main__":
    print("🎓 Aror University Production Chatbot Started!")
    print("🤖 Advanced RAG System Active")

    # Safe check for ai_enabled attribute
    ai_status = "Enabled" if (hasattr(chatbot, 'ai_enabled') and chatbot.ai_enabled) else "Fallback Mode"
    print(f"📊 AI Semantic Search: {ai_status}")
    print(f"📁 Frontend available: {os.path.exists(frontend_dir)}")
    print("🌐 Server running on: http://localhost:8000")
    print("🔍 Health check: http://localhost:8000/health")
    print("🗺️ Available routes: http://localhost:8000/routes")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=True
    )
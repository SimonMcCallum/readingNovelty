# Deployment Guide for PDF Novelty Detection Server

## Quick Start (Development)

1. **Clone and install dependencies**
   ```bash
   git clone https://github.com/SimonMcCallum/readingNovelty.git
   cd readingNovelty
   pip install -r requirements.txt
   ```

2. **Configure API keys**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   nano .env
   ```

3. **Run the server**
   ```bash
   python server.py
   ```

4. **Test the server**
   ```bash
   # Health check
   curl http://localhost:5000/health
   
   # Upload a PDF
   curl -X POST -F "file=@document.pdf" http://localhost:5000/upload
   ```

## Production Deployment

### Using Gunicorn (Recommended)

1. **Install Gunicorn**
   ```bash
   pip install gunicorn
   ```

2. **Run with Gunicorn**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 server:app
   ```

### Using Docker

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   RUN mkdir -p uploads
   
   EXPOSE 5000
   
   CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "server:app"]
   ```

2. **Build and run**
   ```bash
   docker build -t pdf-novelty-server .
   docker run -p 5000:5000 -e ANTHROPIC_API_KEY=your_key pdf-novelty-server
   ```

### Environment Variables

Required environment variables:
- `ANTHROPIC_API_KEY` - Your Anthropic API key (for Claude)
- OR `OPENAI_API_KEY` - Your OpenAI API key (for ChatGPT)

Optional:
- `FLASK_ENV` - Environment (development/production)
- `FLASK_DEBUG` - Enable debug mode (0/1)
- `UPLOAD_FOLDER` - Directory for uploads (default: uploads)
- `MAX_CONTENT_LENGTH` - Max upload size in bytes (default: 16MB)
- `PORT` - Server port (default: 5000)

### Nginx Reverse Proxy

Example Nginx configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    client_max_body_size 20M;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## System Requirements

- Python 3.8 or higher
- 2GB RAM minimum (4GB+ recommended)
- 5GB disk space for models and temporary files
- Network access for downloading models on first run

## Performance Notes

- First request will be slower as models are downloaded and cached
- Subsequent requests use cached models
- Processing time depends on PDF size (typically 10-30 seconds per page)
- Consider using a job queue (e.g., Celery) for large documents

## Troubleshooting

### Models not downloading
- Ensure network access to huggingface.co
- Check firewall settings
- Models are cached in `~/.cache/huggingface/`

### Out of memory errors
- Increase system RAM
- Reduce PDF size or split into smaller documents
- Use a smaller embedding model

### API key errors
- Verify API keys are correctly set in .env
- Check API key has sufficient quota
- Server falls back to simple mode if no keys are available

## Security Considerations

- Keep API keys secure and never commit them to git
- Use HTTPS in production
- Implement rate limiting for public deployments
- Regularly update dependencies
- Consider implementing user authentication
- Sanitize file uploads and set size limits

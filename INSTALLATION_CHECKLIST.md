# Installation & Verification Checklist

## ✅ Pre-Installation Checklist

### System Requirements
- [ ] Python 3.8+ installed
- [ ] MySQL 5.7+ or 8.0+ installed and running
- [ ] CUDA 11.0+ (optional, for GPU acceleration)
- [ ] 4GB+ RAM available
- [ ] 10GB+ disk space available

### Dependencies
- [ ] PyTorch 2.0+ installed
- [ ] OpenCV installed
- [ ] Flask installed
- [ ] MySQL connector installed
- [ ] All requirements from `backend/requirements.txt` installed

## ✅ Installation Steps

### 1. Clone Repository
```bash
cd d:\
git clone <repository-url> LandCoverAI
cd LandCoverAI
```
- [ ] Repository cloned successfully

### 2. Create Virtual Environment
```bash
python -m venv .venv
.\.venv\Scripts\Activate
```
- [ ] Virtual environment created
- [ ] Virtual environment activated

### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```
- [ ] All dependencies installed without errors
- [ ] PyTorch with CUDA support (if GPU available)

### 4. Configure Database
```bash
# Set environment variables
$env:DB_HOST = "localhost"
$env:DB_USER = "root"
$env:DB_PASSWORD = "admin"
$env:DB_NAME = "landcover_db"
```
- [ ] MySQL server running
- [ ] Database credentials configured
- [ ] Database will be auto-created on first run

### 5. Download Model Checkpoints
- [ ] `best_model.pth` in repository root (UNet++ segmentation model)
- [ ] `satellite_classifier_v4.pth` in repository root (satellite gate)
- [ ] Model files are valid PyTorch checkpoints

### 6. Optional: Configure ArcGIS API Key
```bash
$env:ARCGIS_API_KEY = "your_api_key_here"
```
- [ ] ArcGIS API key configured (optional, for heavy usage)

## ✅ Verification Steps

### 1. Test Crop Recommendation Engine
```bash
cd backend
python test_recommender.py
```

Expected output:
- [ ] Test 1: Balanced Agricultural Land → SUB_HUMID regime, 10 crops
- [ ] Test 2: Arid/Semi-Arid Region → ARID/SEMI_ARID regime, drought-tolerant crops
- [ ] Test 3: Water-Rich Region → WATER_RICH regime, rice/sugarcane
- [ ] Test 4: Urban Dominated → Status "halted", appropriate message
- [ ] Test 5: Degraded Soil → DEGRADED soil class, nitrogen-fixers recommended

### 2. Start Flask Server
```bash
python backend/app.py
```

Expected output:
- [ ] Server starts without errors
- [ ] Database tables created automatically
- [ ] Segmentation model loaded successfully
- [ ] Satellite gate loaded successfully
- [ ] Server listening on http://localhost:5000

### 3. Test API Endpoints

#### Register User
```bash
curl -X POST http://localhost:5000/api/register \
  -H "Content-Type: application/json" \
  -d '{"name":"Test User","email":"test@example.com","password":"password123"}'
```
- [ ] Returns 201 status
- [ ] Returns success message

#### Login
```bash
curl -X POST http://localhost:5000/api/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'
```
- [ ] Returns 200 status
- [ ] Returns auth token
- [ ] Save token for next steps

#### Test Crop Recommendation
```bash
curl -X POST http://localhost:5000/api/recommend-crops \
  -H "Authorization: Bearer <your_token>" \
  -H "Content-Type: application/json" \
  -d '{"percentages":{"urban_land":5,"agriculture":45,"rangeland":15,"forest":20,"water":10,"barren":5},"top_n":10}'
```
- [ ] Returns 200 status
- [ ] Returns ranked crops
- [ ] Returns indices (MAI, SHI, ASI, CFI, MEI, AFI)
- [ ] Returns water regime, soil class, market class

#### Test Coordinate Prediction
```bash
curl -X POST http://localhost:5000/api/predict/coordinates \
  -H "Authorization: Bearer <your_token>" \
  -H "Content-Type: application/json" \
  -d '{"lat":28.6139,"lon":77.2090,"radius_m":500,"size":512}'
```
- [ ] Returns 200 status
- [ ] Returns satellite image (base64)
- [ ] Returns annotated image (base64)
- [ ] Returns land-cover percentages
- [ ] Returns detections with bounding boxes

### 4. Test Frontend
```bash
# Open browser
http://localhost:5000
```
- [ ] Login page loads
- [ ] Can register new user
- [ ] Can login with credentials
- [ ] Map interface loads
- [ ] Can click on map to analyze location
- [ ] Segmentation results display
- [ ] Crop recommendations display (if integrated)

### 5. Verify Database
```sql
-- Connect to MySQL
mysql -u root -p

USE landcover_db;

-- Check tables
SHOW TABLES;
-- Expected: users, sessions, predictions

-- Check user
SELECT * FROM users;
-- Expected: Test user record

-- Check predictions
SELECT COUNT(*) FROM predictions;
-- Expected: Number of predictions made
```
- [ ] Database exists
- [ ] All tables created
- [ ] User records present
- [ ] Predictions logged

## ✅ File Structure Verification

### Core Files
- [ ] `backend/app.py` (Flask application)
- [ ] `backend/crop_recommender.py` (Recommendation engine)
- [ ] `backend/test_recommender.py` (Test suite)
- [ ] `backend/example_crop_api.py` (API examples)
- [ ] `backend/crop_config.py` (Configuration)
- [ ] `backend/requirements.txt` (Dependencies)

### Model Files
- [ ] `best_model.pth` (Segmentation model)
- [ ] `satellite_classifier_v4.pth` (Gate classifier)

### Documentation Files
- [ ] `README.md` (Main documentation)
- [ ] `CROP_RECOMMENDATION.md` (Technical documentation)
- [ ] `CROP_QUICK_REFERENCE.md` (Quick reference)
- [ ] `IMPLEMENTATION_SUMMARY.md` (Implementation summary)
- [ ] `SYSTEM_ARCHITECTURE.md` (Architecture diagram)
- [ ] `INSTALLATION_CHECKLIST.md` (This file)

### Frontend Files
- [ ] `frontend/index.html`
- [ ] `frontend/login.html`
- [ ] `frontend/register.html`
- [ ] `frontend/css/` directory
- [ ] `frontend/js/` directory

## ✅ Common Issues & Solutions

### Issue 1: MySQL Connection Error
**Symptom**: `mysql.connector.errors.DatabaseError: 2003`

**Solution**:
- [ ] Verify MySQL server is running
- [ ] Check DB_HOST, DB_USER, DB_PASSWORD environment variables
- [ ] Test connection: `mysql -u root -p`

### Issue 2: Model File Not Found
**Symptom**: `FileNotFoundError: best_model.pth not found`

**Solution**:
- [ ] Verify model files are in repository root
- [ ] Check file names match exactly
- [ ] Re-download model files if corrupted

### Issue 3: CUDA Out of Memory
**Symptom**: `RuntimeError: CUDA out of memory`

**Solution**:
- [ ] Reduce batch size (already 1 in this app)
- [ ] Use CPU instead: Set `CUDA_VISIBLE_DEVICES=""`
- [ ] Close other GPU applications

### Issue 4: Import Error
**Symptom**: `ModuleNotFoundError: No module named 'crop_recommender'`

**Solution**:
- [ ] Verify you're in the correct directory
- [ ] Check virtual environment is activated
- [ ] Reinstall dependencies: `pip install -r backend/requirements.txt`

### Issue 5: Port Already in Use
**Symptom**: `OSError: [Errno 98] Address already in use`

**Solution**:
- [ ] Kill existing Flask process
- [ ] Change port: `app.run(port=5001)`
- [ ] Check for other services on port 5000

## ✅ Performance Benchmarks

### Expected Performance (GPU)
- [ ] Segmentation: 50-100ms per image
- [ ] Crop recommendation: <50ms
- [ ] Total latency: <150ms
- [ ] Memory usage: ~2GB GPU, ~500MB RAM

### Expected Performance (CPU)
- [ ] Segmentation: 500-1000ms per image
- [ ] Crop recommendation: <50ms
- [ ] Total latency: <1100ms
- [ ] Memory usage: ~2GB RAM

## ✅ Production Readiness Checklist

### Security
- [ ] Change default SECRET_KEY in production
- [ ] Use strong database passwords
- [ ] Enable HTTPS for production deployment
- [ ] Implement rate limiting
- [ ] Add input validation and sanitization

### Scalability
- [ ] Configure multiple Flask workers (gunicorn)
- [ ] Set up database connection pooling
- [ ] Implement Redis caching for frequent queries
- [ ] Use CDN for static assets
- [ ] Set up load balancer for multiple instances

### Monitoring
- [ ] Set up logging (already configured)
- [ ] Add application monitoring (e.g., Prometheus)
- [ ] Set up error tracking (e.g., Sentry)
- [ ] Configure database monitoring
- [ ] Set up uptime monitoring

### Backup
- [ ] Configure database backups
- [ ] Back up model files
- [ ] Back up prediction history images
- [ ] Document recovery procedures

## ✅ Next Steps

### For Development
- [ ] Read `CROP_RECOMMENDATION.md` for technical details
- [ ] Review `CROP_QUICK_REFERENCE.md` for quick lookups
- [ ] Customize `crop_config.py` for your region
- [ ] Add new crops to knowledge base
- [ ] Tune scoring weights

### For Deployment
- [ ] Set up production database
- [ ] Configure environment variables
- [ ] Set up reverse proxy (nginx)
- [ ] Configure SSL certificates
- [ ] Deploy to cloud (AWS/Azure/GCP)

### For Integration
- [ ] Update frontend to display crop recommendations
- [ ] Add crop recommendation to prediction history
- [ ] Implement crop comparison features
- [ ] Add export functionality (PDF/CSV)
- [ ] Integrate with external APIs (weather, prices)

## ✅ Support & Resources

### Documentation
- Main README: `README.md`
- Technical Guide: `CROP_RECOMMENDATION.md`
- Quick Reference: `CROP_QUICK_REFERENCE.md`
- Architecture: `SYSTEM_ARCHITECTURE.md`

### Testing
- Test Suite: `backend/test_recommender.py`
- API Examples: `backend/example_crop_api.py`

### Configuration
- Crop Config: `backend/crop_config.py`
- Environment Variables: See README.md

### Community
- Report issues on GitHub
- Contribute improvements via pull requests
- Share feedback and suggestions

---

## Final Verification

All checks passed? ✅

- [ ] Installation complete
- [ ] All tests passing
- [ ] API endpoints working
- [ ] Frontend accessible
- [ ] Database configured
- [ ] Documentation reviewed

**System Status**: Ready for use! 🚀

---

**Last Updated**: 2024
**Version**: 1.0.0

# Frontend Integration Guide — Crop Recommendations

## Overview

This guide shows how to integrate crop recommendations with explanations into your frontend application.

## API Response Structure

### Complete Response with Explanations

```json
{
  "status": "ok",
  "halt_message": null,
  "flags": [],
  "water_regime": "SUB_HUMID",
  "secondary_regime": null,
  "soil_class": "MODERATE",
  "market_class": "SUBSISTENCE",
  "indices": {
    "MAI": 37.0,
    "SHI": 51.0,
    "ASI": 3.0,
    "CFI": 47.0,
    "MEI": 18.5,
    "AFI": 24.5
  },
  "signals": {
    "urban": 5.0,
    "agri": 45.0,
    "range": 15.0,
    "forest": 20.0,
    "water": 10.0,
    "barren": 5.0
  },
  "ranked_crops": [
    {
      "rank": 1,
      "crop": "Wheat",
      "category": "Cereal",
      "season": "Rabi",
      "score": 78.5,
      "regime_match": true,
      "marginal": false,
      "agroforestry": false
    }
  ],
  "explanations": {
    "summary": "✅ Analysis complete! ...",
    "land_analysis": "📊 Land Cover Distribution...",
    "indices_explanation": "📈 Environmental Indices...",
    "regime_explanation": "☁️ Water Regime: Sub Humid...",
    "soil_explanation": "⚠️ Soil Class: Moderate...",
    "market_explanation": "🏘️ Market Class: Subsistence...",
    "crop_explanations": [
      {
        "crop": "Wheat",
        "rank": 1,
        "score": 78.5,
        "confidence": "HIGH",
        "reasoning": "✅ Perfect match for sub humid climate • ✅ Highly suitable...",
        "pros": ["High success probability", "Climate-appropriate"],
        "cons": [],
        "requirements": {
          "season": "Rabi",
          "category": "Cereal",
          "agroforestry_compatible": false
        }
      }
    ],
    "recommendations_summary": "🎯 Actionable Recommendations..."
  }
}
```

## JavaScript Integration Example

### 1. Fetch Crop Recommendations

```javascript
async function getCropRecommendations(landcoverPercentages) {
  const response = await fetch('/api/recommend-crops', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${authToken}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      percentages: landcoverPercentages,
      top_n: 10,
      include_explanations: true
    })
  });
  
  if (!response.ok) {
    throw new Error('Failed to get crop recommendations');
  }
  
  return await response.json();
}
```

### 2. Display Summary

```javascript
function displaySummary(result) {
  const summaryDiv = document.getElementById('crop-summary');
  
  if (result.status === 'halted') {
    summaryDiv.innerHTML = `
      <div class="alert alert-warning">
        <h4>⚠️ Analysis Halted</h4>
        <p>${result.halt_message}</p>
      </div>
    `;
    return;
  }
  
  summaryDiv.innerHTML = `
    <div class="summary-card">
      <h3>Analysis Summary</h3>
      <p>${result.explanations.summary}</p>
      
      <div class="classification-badges">
        <span class="badge badge-primary">
          ${result.water_regime.replace('_', ' ')}
        </span>
        <span class="badge badge-success">
          ${result.soil_class}
        </span>
        <span class="badge badge-info">
          ${result.market_class.replace('_', ' ')}
        </span>
      </div>
      
      ${result.flags.length > 0 ? `
        <div class="alert alert-info mt-3">
          <strong>⚠️ Important Notes:</strong>
          <ul>
            ${result.flags.map(flag => `<li>${flag}</li>`).join('')}
          </ul>
        </div>
      ` : ''}
    </div>
  `;
}
```

### 3. Display Land Cover Analysis

```javascript
function displayLandCoverAnalysis(explanations) {
  const analysisDiv = document.getElementById('land-analysis');
  
  // Convert markdown-style text to HTML
  const htmlContent = explanations.land_analysis
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
  
  analysisDiv.innerHTML = `
    <div class="analysis-card">
      ${htmlContent}
    </div>
  `;
}
```

### 4. Display Environmental Indices

```javascript
function displayIndices(indices, explanations) {
  const indicesDiv = document.getElementById('indices-display');
  
  const indexData = [
    { name: 'MAI', value: indices.MAI, label: 'Moisture Availability', color: '#3498db' },
    { name: 'SHI', value: indices.SHI, label: 'Soil Health', color: '#2ecc71' },
    { name: 'ASI', value: indices.ASI, label: 'Aridity Stress', color: '#e74c3c' },
    { name: 'CFI', value: indices.CFI, label: 'Cultivation Feasibility', color: '#f39c12' },
    { name: 'MEI', value: indices.MEI, label: 'Market Access', color: '#9b59b6' },
    { name: 'AFI', value: indices.AFI, label: 'Agroforestry Potential', color: '#1abc9c' }
  ];
  
  indicesDiv.innerHTML = `
    <div class="indices-grid">
      ${indexData.map(index => `
        <div class="index-card">
          <h4>${index.label}</h4>
          <div class="progress-bar-container">
            <div class="progress-bar" style="width: ${index.value}%; background-color: ${index.color}">
              ${index.value.toFixed(1)}
            </div>
          </div>
        </div>
      `).join('')}
    </div>
    
    <div class="indices-explanation mt-3">
      ${explanations.indices_explanation.replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}
    </div>
  `;
}
```

### 5. Display Crop Recommendations

```javascript
function displayCropRecommendations(cropExplanations) {
  const cropsDiv = document.getElementById('crop-recommendations');
  
  cropsDiv.innerHTML = `
    <div class="crops-list">
      ${cropExplanations.map(crop => `
        <div class="crop-card ${getConfidenceClass(crop.confidence)}">
          <div class="crop-header">
            <h3>
              ${getConfidenceEmoji(crop.confidence)} 
              ${crop.rank}. ${crop.crop}
            </h3>
            <span class="crop-score">Score: ${crop.score.toFixed(1)}/100</span>
          </div>
          
          <div class="crop-meta">
            <span class="badge">${crop.requirements.category}</span>
            <span class="badge">${crop.requirements.season}</span>
            ${crop.requirements.agroforestry_compatible ? 
              '<span class="badge badge-success">🌳 Agroforestry</span>' : ''}
          </div>
          
          <div class="crop-reasoning">
            <p>${crop.reasoning}</p>
          </div>
          
          <div class="crop-details">
            <div class="pros">
              <strong>✅ Advantages:</strong>
              <ul>
                ${crop.pros.map(pro => `<li>${pro}</li>`).join('')}
              </ul>
            </div>
            
            ${crop.cons.length > 0 ? `
              <div class="cons">
                <strong>⚠️ Challenges:</strong>
                <ul>
                  ${crop.cons.map(con => `<li>${con}</li>`).join('')}
                </ul>
              </div>
            ` : ''}
          </div>
          
          <button class="btn btn-primary btn-sm" onclick="showCropDetails('${crop.crop}')">
            More Details
          </button>
        </div>
      `).join('')}
    </div>
  `;
}

function getConfidenceClass(confidence) {
  const classes = {
    'HIGH': 'confidence-high',
    'MODERATE': 'confidence-moderate',
    'LOW': 'confidence-low'
  };
  return classes[confidence] || '';
}

function getConfidenceEmoji(confidence) {
  const emojis = {
    'HIGH': '✅',
    'MODERATE': '⚠️',
    'LOW': '🔶'
  };
  return emojis[confidence] || '📍';
}
```

### 6. Display Recommendations Summary

```javascript
function displayRecommendationsSummary(explanations) {
  const summaryDiv = document.getElementById('recommendations-summary');
  
  const htmlContent = explanations.recommendations_summary
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
  
  summaryDiv.innerHTML = `
    <div class="recommendations-card">
      <h3>🎯 Actionable Recommendations</h3>
      ${htmlContent}
    </div>
  `;
}
```

### 7. Complete Integration Function

```javascript
async function showCropRecommendations(landcoverPercentages) {
  try {
    // Show loading state
    showLoading();
    
    // Fetch recommendations
    const result = await getCropRecommendations(landcoverPercentages);
    
    // Display all sections
    displaySummary(result);
    displayLandCoverAnalysis(result.explanations);
    displayIndices(result.indices, result.explanations);
    displayCropRecommendations(result.explanations.crop_explanations);
    displayRecommendationsSummary(result.explanations);
    
    // Show regime and soil details
    displayRegimeDetails(result.explanations);
    displaySoilDetails(result.explanations);
    displayMarketDetails(result.explanations);
    
    hideLoading();
  } catch (error) {
    console.error('Error fetching crop recommendations:', error);
    showError('Failed to load crop recommendations. Please try again.');
  }
}
```

## CSS Styling Example

```css
/* Crop Recommendation Styles */

.crop-card {
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 20px;
  transition: all 0.3s ease;
}

.crop-card:hover {
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  transform: translateY(-2px);
}

.confidence-high {
  border-left: 4px solid #2ecc71;
}

.confidence-moderate {
  border-left: 4px solid #f39c12;
}

.confidence-low {
  border-left: 4px solid #e74c3c;
}

.crop-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.crop-score {
  font-size: 1.2em;
  font-weight: bold;
  color: #3498db;
}

.crop-meta {
  display: flex;
  gap: 10px;
  margin-bottom: 15px;
}

.badge {
  padding: 5px 10px;
  border-radius: 4px;
  font-size: 0.85em;
  background-color: #ecf0f1;
  color: #2c3e50;
}

.badge-success {
  background-color: #2ecc71;
  color: white;
}

.crop-reasoning {
  background-color: #f8f9fa;
  padding: 15px;
  border-radius: 4px;
  margin-bottom: 15px;
}

.crop-details {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-top: 15px;
}

.pros, .cons {
  padding: 10px;
  border-radius: 4px;
}

.pros {
  background-color: #d4edda;
}

.cons {
  background-color: #fff3cd;
}

.indices-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 20px;
}

.index-card {
  background: white;
  padding: 15px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.progress-bar-container {
  width: 100%;
  height: 30px;
  background-color: #ecf0f1;
  border-radius: 4px;
  overflow: hidden;
  margin-top: 10px;
}

.progress-bar {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: bold;
  transition: width 0.5s ease;
}

.classification-badges {
  display: flex;
  gap: 10px;
  margin-top: 15px;
}

.recommendations-card {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 30px;
  border-radius: 12px;
  margin-top: 30px;
}
```

## Complete Workflow

1. **User uploads image or selects coordinates**
2. **Segmentation runs** → Get land-cover percentages
3. **Call crop recommendation API** with percentages
4. **Display results** in organized sections:
   - Summary with status badges
   - Land cover distribution
   - Environmental indices with progress bars
   - Top crop recommendations with reasoning
   - Actionable recommendations

## Error Handling

```javascript
function handleCropRecommendationError(error) {
  if (error.status === 422) {
    // Halted recommendation
    showWarning(error.halt_message);
  } else if (error.status === 401) {
    // Unauthorized
    redirectToLogin();
  } else {
    // General error
    showError('Failed to generate crop recommendations. Please try again.');
  }
}
```

## Performance Optimization

- Cache crop recommendations for same land-cover percentages
- Lazy load detailed explanations
- Use pagination for large crop lists
- Implement progressive disclosure (show summary first, details on demand)

## Accessibility

- Use semantic HTML
- Add ARIA labels for screen readers
- Ensure color contrast meets WCAG standards
- Provide text alternatives for emojis
- Support keyboard navigation

## Mobile Responsiveness

```css
@media (max-width: 768px) {
  .crop-details {
    grid-template-columns: 1fr;
  }
  
  .indices-grid {
    grid-template-columns: 1fr;
  }
  
  .crop-header {
    flex-direction: column;
    align-items: flex-start;
  }
}
```

This integration provides a complete, user-friendly interface for displaying crop recommendations with transparent reasoning and actionable insights!

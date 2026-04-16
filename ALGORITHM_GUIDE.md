# LandCoverAI: The Core Six Algorithms

This guide documents the entire LandCoverAI application as a set of six primary algorithms. These algorithms represent the "logical engine" that drives everything from user security to high-precision agricultural recommendations.

---

## 1. Identity & Access Management (IAM)
**Purpose**: Secures user data and provides persistent sessions across the platform.

### REGISTER:
*   **INPUT**: `name`, `email`, `password`
*   **LOGIC**:
    1.  **VALIDATE**: Check email format and password length requirements.
    2.  **HASH**: Generate a secure salt and compute `hash ← bcrypt.hashpw(password)`.
    3.  **INSERT**: Write `(name, email, hash, now())` into the `users` table.
*   **OUTPUT**: `201 Created` (Success) or `409 Conflict` (if email exists).

### LOGIN:
*   **INPUT**: `email`, `password`
*   **LOGIC**:
    1.  **QUERY**: Fetch the user record by `email`.
    2.  **VERIFY**: If user not found OR `bcrypt.checkpw(password, hash)` is `false` → **RETURN** `401 Unauthorized`.
    3.  **SESSION**: Generate a secure `token ← secrets.token_urlsafe(32)`.
    4.  **PERSIST**: Insert `(token, user_id, now())` into the `sessions` table.
*   **OUTPUT**: `{ token, user_profile }`.

### MIDDLEWARE (Protected Routes):
*   **INPUT**: HTTP Header `Authorization: Bearer <token>`
*   **LOGIC**:
    1.  **EXTRACT**: Parse the Bearer token from the request header.
    2.  **VERIFY**: Perform a SQL `JOIN` between `users` and `sessions` where `token` matches.
    3.  **ATTACH**: If match found, attach `current_user` object to the request context.
    4.  **REJECT**: If no match or token expired → **RETURN** `401 Unauthorized`.

---

## 2. Dynamic Multilingual Engine (i18n)
**Purpose**: Provides a localized experience across 6+ languages (Telugu, Hindi, Tamil, etc.).

*   **INPUT**: `target_lang_code`
*   **LOGIC**:
    1.  **FETCH**: Load the translation JSON dictionary (e.g., `window._i18nData`).
    2.  **SET**: Update `localStorage` and global `currentLang` variable.
    3.  **SCAN**: Iterate through all DOM elements with `data-i18n` attributes.
    4.  **REPLACE**:
        -   If element is text: `element.textContent = data[key][lang]`.
        -   If element has placeholder: `element.placeholder = data[key][lang]`.
*   **OUTPUT**: Fully localized UI without page refresh.

---

## 3. High-Precision Satellite Vision Pipeline
**Purpose**: Transforms raw satellite tiles into quantified land-cover data.

*   **INPUT**: `latitude`, `longitude` (or uploaded image)
*   **LOGIC**:
    1.  **TILE FETCH**: Request a 512x512 tile from the ArcGIS Imagery Service at the specified coordinates.
    2.  **GATE CHECK**: Pass tile through **EfficientNet-B0** binary classifier.
        -   If `score < threshold` → **REJECT** (Image is not a satellite view).
    3.  **SEGMENTATION**: Pass tile through **UNet++ (EfficientNet-B7 encoder)**.
    4.  **QUANTIFICATION**:
        -   Iterate through each of the 512x512 pixels.
        -   Assign each pixel to one of 7 classes (Urban, Agri, Forest, etc.).
        -   **CALCULATE**: `class_percentage = (pixel_count / total_pixels) * 100`.
*   **OUTPUT**: `land_cover_vector` (e.g., `{agri: 65, forest: 15, water: 5, ...}`).

---

## 4. Contextual Suitability Scoring
**Purpose**: Ranks crops based on their affinity for the detected landscape.

*   **INPUT**: `land_cover_vector`, `target_crop_profile`
*   **LOGIC**:
    1.  **EUCLIDEAN DIST**: Calculate the weighted distance between `observed_vector` and `ideal_crop_vector`.
    2.  **TERRAIN ARCHETYPE**:
        -   If `water > 10%` → Apply **Paddy Zone** bonus (+15 pts for Rice/Jute).
        -   If `forest > 15%` → Apply **Agroforestry** bonus (+12 pts for Coffee/Spice).
    3.  **MARGINAL BLEND**: If `agriculture > 70%`, amplify the weights of non-ag features (barren/forest) to avoid "generic" results.
*   **OUTPUT**: `suitability_score` (0-100).

---

## 5. Diversity-Aware Ranking (MMR)
**Purpose**: Prevents the "15 Cereals" problem by ensuring diversity in the recommendation list.

*   **INPUT**: List of 100 scored crops.
*   **LOGIC**:
    1.  **POOL**: Sort all crops by `suitability_score` descending.
    2.  **SELECT TOP-1**: The highest-scoring crop is always selected.
    3.  **ITERATIVE SELECTION**: For remaining slots (up to 15):
        -   Calculate `MMR Score = λ * Relevance - (1-λ) * Redundancy`.
        -   **Relevance**: The crop's suitability score.
        -   **Redundancy**: Maximum similarity to any crop already in the "Selected" list.
        -   **CATEGORY CAP**: Subtract penalty if a category (e.g., "Cereal") is already over-represented.
*   **OUTPUT**: 15 varied, high-precision crop recommendations.

---

## 6. Result Persistence & Archival
**Purpose**: Provides deterministic history and efficient storage of analyses.

*   **INPUT**: `segmentation_results`, `original_image`, `user_id`
*   **LOGIC**:
    1.  **UUID**: Generate a unique `UUID` for the prediction entry.
    2.  **FILE IO**: Save the original image and the annotated "mask" image to disk.
    3.  **COMPACT**: Remove large base64 blobs from the JSON result and convert to a compact `results_json`.
    4.  **SQL INSERT**: Write `(user_id, orig_path, ann_path, results_json, created_at)` to the `predictions` table.
*   **OUTPUT**: Persistent record accessible via the "History" dashboard.

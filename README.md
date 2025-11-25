# applied-ai
# 📦 Amazon Bin Inventory Verifier

An internal-facing Streamlit application for **verifying Amazon-style bin contents** against an expected order using a stack of computer vision and multimodal models (OWL-v2, YOLOv8, and Gemini 2.5 Pro).

The app is designed to act like a lightweight **“bin QA workstation”**: drop in a photo of a bin, key in the ASINs and quantities, and get an automated pass/fail verdict with model-specific insights.

---

## 📖 How to Use

### 🖼️ 1. Upload a Bin Image
1. Navigate to the **Upload Bin Image** section.
2. **Drag & drop** a `.jpg`, `.jpeg`, or `.png` file into the upload box.
3. The image will appear immediately in the preview window.
4. **Auto-Count:** The app automatically estimates the item count using an internal AI model (displayed below the image).
   > *Tip: Use this count as a quick visual + AI sanity check before proceeding.*

### 📦 2. Enter Items to Verify
On the right-side panel under **Verify Order**:

1. For each item, enter:
   * **ASIN**
   * **Quantity**
2. *Note:* A product name will automatically appear if the ASIN is recognized.
3. Use the **"Add Another Item"** button to include more products.
4. Use the **trash-icon** to remove entries.
   *This establishes the "expected order" ground truth.*

### 🤖 3. Choose the Verification Model
Select one of three analysis modes from the left sidebar:

* **General Purpose (OWL-v2)**
   * Flexible, open-vocabulary detection.
   * *Best for:* Varied product packaging.
* **High Precision (YOLOv8)**
   * High-accuracy detection.
   * *Best for:* Known and consistent item types.
* **Gemini 2.5 Pro (Backup)**
   * Google’s multimodal AI.
   * Performs reasoning based on Image + Order context.
   * *Best for:* Natural-language verification and complex scenarios.

> **Settings:** Adjust the **Confidence Threshold** slider for computer vision models (OWL/YOLO).
> * **Lower:** More detections (risk of false positives).
> * **Higher:** Stricter matching.

### 🔍 4. Verify the Order
Once inputs are set:

1. Click the **Verify Order** button.
2. Results will render directly under the order form.

| Model Type | Output Description |
| :--- | :--- |
| **OWL-v2 / YOLOv8** | • List of detected items<br>• Found vs. Expected quantities<br>• PASS/MISMATCH status per ASIN<br>• **(YOLO Only)** Annotated image with bounding boxes |
| **Gemini 2.5 Pro** | • Human-readable inspection report<br>• Explains visible items<br>• Details match/mismatch reasoning |

### 🧾 5. Interpreting Results
The app provides a clear breakdown of the analysis:

* ✔️ **Item verified:** Correct quantity found.
* ❌ **Mismatch:** Fewer items detected than expected.
* ⚠️ **Item not detected:** The item is missing entirely from the view.
* 📝 **Gemini reasoning:** Descriptive explanation of the findings.

---

## 🎯 Summary
The workflow is designed for speed and accuracy:
1. **Upload** bin image.
2. **Enter** expected ASINs & quantities.
3. **Pick** an AI model.
4. **Get** automated verification report.


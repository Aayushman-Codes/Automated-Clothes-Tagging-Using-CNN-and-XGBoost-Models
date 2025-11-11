import React, { useState } from "react";

export default function Predictor() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [price, setPrice] = useState<number | null>(null);
  const [cnnOutput, setCnnOutput] = useState<number[][] | null>(null);

  const backendURL = import.meta.env.VITE_API_URL || "http://localhost:5000";

  const handleUpload = async () => {
    if (!file) {
      alert("Please select an image first!");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    // Example product metadata — you can later make this dynamic
    const metadata = {
      product_type_name: "dress",
      colour_group_name: "Black",
      garment_group_name: "Jersey Basic",
    };
    formData.append("metadata", JSON.stringify(metadata));

    try {
      const res = await fetch(`${backendURL}/predict`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setPrice(data.predicted_price);
      setCnnOutput(data.cnn_output);
    } catch (err) {
      console.error("Prediction failed:", err);
      alert("Prediction failed. Check console for details.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 border rounded-xl shadow-md bg-white max-w-lg mx-auto">
      <h2 className="text-lg font-semibold mb-4">Upload Image for Prediction</h2>

      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        className="mb-4"
      />

      <button
        onClick={handleUpload}
        disabled={!file || loading}
        className={`px-4 py-2 rounded text-white ${
          loading ? "bg-gray-400" : "bg-blue-600 hover:bg-blue-700"
        }`}
      >
        {loading ? "Predicting..." : "Predict"}
      </button>

      {price !== null && (
        <div className="mt-4 text-green-600 font-medium">
          Predicted Price: ₹{price.toFixed(2)}
        </div>
      )}

      {cnnOutput && (
        <pre className="mt-2 text-sm bg-gray-50 p-2 rounded">
          CNN Output: {JSON.stringify(cnnOutput[0].slice(0, 10), null, 2)}...
        </pre>
      )}
    </div>
  );
}

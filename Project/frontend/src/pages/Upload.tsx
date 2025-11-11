import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Upload as UploadIcon, Image as ImageIcon } from "lucide-react";
import Navigation from "@/components/Navigation";
import { toast } from "sonner";

const Upload = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const backendURL =
    import.meta.env.VITE_API_URL?.trim() || "http://localhost:5000";

  // Handle drag & drop
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const handleFile = (file: File) => {
    if (!file.type.startsWith("image/")) {
      toast.error("Please upload an image file (JPG, PNG, GIF)");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      toast.error("File size must be less than 10MB");
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => setSelectedImage(e.target?.result as string);
    reader.readAsDataURL(file);
    setFile(file);
  };

  // 🔥 Send image to Flask backend
  const handleViewResults = async () => {
    if (!file) {
      toast.error("Please upload an image first!");
      return;
    }

    try {
      setLoading(true);
      toast.message("Processing image...", {
        description: "Running CNN + XGBoost model...",
      });

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${backendURL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (data.error) throw new Error(data.error);

      // ✅ Save predictions in localStorage for Results page
      localStorage.setItem("uploadedImage", selectedImage!);
      localStorage.setItem("predictedProduct", data.predicted_product ?? "N/A");
      localStorage.setItem("predictedColor", data.predicted_color ?? "N/A");
      localStorage.setItem(
        "predictedPrice",
        data.predicted_price ? data.predicted_price.toFixed(2) : "N/A"
      );

      // Optional: store top-5 predictions for better insights
      localStorage.setItem(
        "topProducts",
        JSON.stringify(data.topk_product ?? [])
      );
      localStorage.setItem("topColors", JSON.stringify(data.topk_color ?? []));

      toast.success("Prediction complete!");
      navigate("/results");
    } catch (err: any) {
      console.error("Prediction failed:", err);
      toast.error("Prediction failed. Please check the console for details.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <main className="container mx-auto px-4 pt-32 pb-20">
        <div className="max-w-3xl mx-auto space-y-8 animate-fade-in">
          <div className="text-center space-y-4">
            <h1 className="text-4xl md:text-5xl font-bold">
              Upload Clothing Image
            </h1>
            <p className="text-xl text-muted-foreground">
              Drag and drop or select an image to get instant AI predictions
            </p>
          </div>

          <Card className="p-8 shadow-soft">
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`relative border-2 border-dashed rounded-xl p-12 transition-all duration-300 ${
                isDragging
                  ? "border-primary bg-primary/5 scale-105"
                  : "border-border bg-muted/30 hover:border-primary/50"
              }`}
            >
              <input
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />

              <div className="flex flex-col items-center gap-4 text-center">
                {selectedImage ? (
                  <>
                    <img
                      src={selectedImage}
                      alt="Selected"
                      className="max-h-64 rounded-lg shadow-soft"
                    />
                    <p className="text-sm text-muted-foreground">
                      Click to change image
                    </p>
                  </>
                ) : (
                  <>
                    <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                      <UploadIcon className="w-8 h-8 text-primary" />
                    </div>
                    <div>
                      <p className="text-lg font-medium">
                        Drop your image here or click to browse
                      </p>
                      <p className="text-sm text-muted-foreground mt-2">
                        Supports JPG, PNG, GIF (max 10MB)
                      </p>
                    </div>
                  </>
                )}
              </div>
            </div>

            {selectedImage && (
              <Button
                onClick={handleViewResults}
                size="lg"
                className="w-full mt-6 shadow-soft hover:shadow-hover transition-all duration-300"
                disabled={loading}
              >
                <ImageIcon className="w-5 h-5 mr-2" />
                {loading ? "Predicting..." : "View Results"}
              </Button>
            )}
          </Card>
        </div>
      </main>
    </div>
  );
};

export default Upload;

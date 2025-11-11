import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ThumbsUp, Sparkles } from "lucide-react";
import Navigation from "@/components/Navigation";
import { toast } from "sonner";

interface TopPrediction {
  index: number;
  class: string | null;
  score: number;
}

const Results = () => {
  const [image, setImage] = useState<string | null>(null);
  const [product, setProduct] = useState<string>("");
  const [color, setColor] = useState<string>("");
  const [price, setPrice] = useState<string>("₹0.00");
  const [topProducts, setTopProducts] = useState<TopPrediction[]>([]);
  const [topColors, setTopColors] = useState<TopPrediction[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    const savedImage = localStorage.getItem("uploadedImage");
    const savedProduct = localStorage.getItem("predictedProduct");
    const savedColor = localStorage.getItem("predictedColor");
    const savedPrice = localStorage.getItem("predictedPrice");
    const savedTopProducts = localStorage.getItem("topProducts");
    const savedTopColors = localStorage.getItem("topColors");

    if (!savedImage || !savedProduct || !savedColor || !savedPrice) {
      toast.error("No prediction data found. Please upload an image first.");
      navigate("/upload");
      return;
    }

    setImage(savedImage);
    setProduct(savedProduct);
    setColor(savedColor);
    setPrice(savedPrice.startsWith("₹") ? savedPrice : `₹${savedPrice}`);

    if (savedTopProducts) setTopProducts(JSON.parse(savedTopProducts));
    if (savedTopColors) setTopColors(JSON.parse(savedTopColors));
  }, [navigate]);

  const handleRateClick = () => {
    navigate("/feedback");
  };

  if (!image) return null;

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <main className="container mx-auto px-4 pt-32 pb-20">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Header */}
          <div className="text-center space-y-4 animate-fade-in">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-muted border border-border">
              <Sparkles className="w-4 h-4 text-primary" />
              <span className="text-sm font-medium text-muted-foreground">
                AI Analysis Complete
              </span>
            </div>
            <h1 className="text-4xl md:text-5xl font-bold">AI Predictions</h1>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {/* Uploaded Image */}
            <Card className="p-6 shadow-soft animate-scale-in">
              <img
                src={image}
                alt="Uploaded clothing"
                className="w-full rounded-lg"
              />
            </Card>

            {/* Prediction Results */}
            <div className="space-y-6 animate-slide-up">
              {/* Product */}
              <Card className="p-6 shadow-soft">
                <h2 className="text-2xl font-bold mb-3">Predicted Product</h2>
                <p className="text-xl text-muted-foreground mb-4">{product}</p>

                {topProducts.length > 0 && (
                  <div className="space-y-1">
                    <p className="text-sm font-medium text-muted-foreground mb-1">
                      Top Product Predictions:
                    </p>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      {topProducts.map((p, idx) => (
                        <li key={idx}>
                          {p.class} — {(p.score * 100).toFixed(1)}%
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </Card>

              {/* Color */}
              <Card className="p-6 shadow-soft">
                <h2 className="text-2xl font-bold mb-3">Predicted Color</h2>
                <p className="text-xl text-muted-foreground mb-4">{color}</p>

                {topColors.length > 0 && (
                  <div className="space-y-1">
                    <p className="text-sm font-medium text-muted-foreground mb-1">
                      Top Color Predictions:
                    </p>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      {topColors.map((c, idx) => (
                        <li key={idx}>
                          {c.class} — {(c.score * 100).toFixed(1)}%
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </Card>

              {/* Price */}
              <Card className="p-6 shadow-soft bg-gradient-card">
                <h2 className="text-2xl font-bold mb-2">Predicted Price</h2>
                <div className="text-5xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                  {price}
                </div>
              </Card>

              {/* Info note */}
              <Card className="p-6 shadow-soft bg-muted/30">
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Predictions are generated using a hybrid deep learning model:
                  a PyTorch ResNet50 CNN for product and color classification,
                  combined with an XGBoost regressor for price estimation.
                </p>
              </Card>

              <Button
                onClick={handleRateClick}
                size="lg"
                className="w-full shadow-soft hover:shadow-hover transition-all duration-300"
              >
                <ThumbsUp className="w-5 h-5 mr-2" />
                Give Feedback
              </Button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Results;

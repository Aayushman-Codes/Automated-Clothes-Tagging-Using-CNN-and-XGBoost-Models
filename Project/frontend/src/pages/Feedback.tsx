import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { CheckCircle } from "lucide-react";
import Navigation from "@/components/Navigation";
import { toast } from "sonner";

const Feedback = () => {
  const [rating, setRating] = useState<"perfect" | "good" | "unsatisfactory" | null>(null);
  const [feedback, setFeedback] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const navigate = useNavigate();

  // Protect the feedback page - redirect if no results available
  useEffect(() => {
    const uploadedImage = localStorage.getItem("uploadedImage");
    if (!uploadedImage) {
      navigate("/upload");
    }
  }, [navigate]);

  const handleSubmit = () => {
    if (!rating) {
      toast.error("Please select a rating");
      return;
    }
    
    // In a real app, this would send data to an API
    console.log({ rating, feedback });
    setSubmitted(true);
    toast.success("Thank you for your feedback!");
    
    setTimeout(() => {
      navigate("/");
    }, 2000);
  };

  if (submitted) {
    return (
      <div className="min-h-screen bg-background">
        <Navigation />
        
        <main className="container mx-auto px-4 pt-32 pb-20">
          <div className="max-w-2xl mx-auto text-center space-y-6 animate-scale-in">
            <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center mx-auto">
              <CheckCircle className="w-10 h-10 text-primary" />
            </div>
            <h1 className="text-4xl font-bold">Thank You!</h1>
            <p className="text-xl text-muted-foreground">
              Your feedback helps us improve our AI model.
            </p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <main className="container mx-auto px-4 pt-32 pb-20">
        <div className="max-w-2xl mx-auto space-y-8 animate-fade-in">
          <div className="text-center space-y-4">
            <h1 className="text-4xl md:text-5xl font-bold">
              Rate This Prediction
            </h1>
            <p className="text-xl text-muted-foreground">
              How accurate was the AI's result?
            </p>
          </div>

          <Card className="p-8 shadow-soft space-y-6">
            <div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Button
                  size="lg"
                  variant={rating === "perfect" ? "default" : "outline"}
                  onClick={() => setRating("perfect")}
                  className="h-24 transition-all duration-300 hover:scale-105"
                >
                  <div className="flex flex-col items-center gap-2">
                    <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center">
                      <span className="text-white text-xl">✓</span>
                    </div>
                    <span className="text-lg font-semibold">Perfect</span>
                  </div>
                </Button>
                
                <Button
                  size="lg"
                  variant={rating === "good" ? "default" : "outline"}
                  onClick={() => setRating("good")}
                  className="h-24 transition-all duration-300 hover:scale-105"
                >
                  <div className="flex flex-col items-center gap-2">
                    <div className="w-8 h-8 rounded-full bg-yellow-500 flex items-center justify-center">
                      <span className="text-white text-xl">~</span>
                    </div>
                    <span className="text-lg font-semibold">Good</span>
                  </div>
                </Button>

                <Button
                  size="lg"
                  variant={rating === "unsatisfactory" ? "default" : "outline"}
                  onClick={() => setRating("unsatisfactory")}
                  className="h-24 transition-all duration-300 hover:scale-105"
                >
                  <div className="flex flex-col items-center gap-2">
                    <div className="w-8 h-8 rounded-full bg-red-500 flex items-center justify-center">
                      <span className="text-white text-xl">✗</span>
                    </div>
                    <span className="text-lg font-semibold">Unsatisfactory</span>
                  </div>
                </Button>
              </div>
            </div>

            <div>
              <h2 className="text-xl font-semibold mb-4">Additional Comments (Optional)</h2>
              <Textarea
                placeholder="Tell us more about your experience..."
                value={feedback}
                onChange={(e) => setFeedback(e.target.value)}
                className="min-h-32 resize-none"
              />
            </div>

            <Button
              onClick={handleSubmit}
              size="lg"
              className="w-full shadow-soft hover:shadow-hover transition-all duration-300"
            >
              Submit Feedback
            </Button>

            <Button
              onClick={() => navigate("/")}
              size="lg"
              variant="outline"
              className="w-full transition-all duration-300"
            >
              Back to Home
            </Button>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default Feedback;

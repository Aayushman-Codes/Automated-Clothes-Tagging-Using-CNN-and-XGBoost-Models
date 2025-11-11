import Navigation from "@/components/Navigation";
import { Brain, Camera, Tags, TrendingUp } from "lucide-react";

const LearnMore = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <main className="container mx-auto px-4 pt-32 pb-20">
        <div className="max-w-4xl mx-auto space-y-12 animate-fade-in">
          <div className="text-center space-y-4">
            <h1 className="text-4xl md:text-5xl font-bold">
              How It{" "}
              <span className="bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                Works
              </span>
            </h1>
            <p className="text-xl text-muted-foreground">
              Understanding the technology behind automated product tagging
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            <div className="space-y-4 p-6 rounded-xl border border-border bg-card hover:shadow-hover transition-all duration-300">
              <div className="w-12 h-12 rounded-lg bg-gradient-hero flex items-center justify-center">
                <Brain className="w-6 h-6 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-semibold">Machine Learning</h3>
              <p className="text-muted-foreground">
                This system leverages machine learning and computer vision to process and interpret clothing images.
              </p>
            </div>

            <div className="space-y-4 p-6 rounded-xl border border-border bg-card hover:shadow-hover transition-all duration-300">
              <div className="w-12 h-12 rounded-lg bg-gradient-hero flex items-center justify-center">
                <Camera className="w-6 h-6 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-semibold">Image Analysis</h3>
              <p className="text-muted-foreground">
                Uploaded images are analyzed to extract visual attributes such as color, texture, and pattern.
              </p>
            </div>

            <div className="space-y-4 p-6 rounded-xl border border-border bg-card hover:shadow-hover transition-all duration-300">
              <div className="w-12 h-12 rounded-lg bg-gradient-hero flex items-center justify-center">
                <Tags className="w-6 h-6 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-semibold">Tag Generation</h3>
              <p className="text-muted-foreground">
                The model then generates descriptive product tags and predicts an estimated price based on learned patterns.
              </p>
            </div>

            <div className="space-y-4 p-6 rounded-xl border border-border bg-card hover:shadow-hover transition-all duration-300">
              <div className="w-12 h-12 rounded-lg bg-gradient-hero flex items-center justify-center">
                <TrendingUp className="w-6 h-6 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-semibold">Continuous Improvement</h3>
              <p className="text-muted-foreground">
                User feedback on prediction accuracy is collected to support model evaluation and continuous improvement.
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default LearnMore;

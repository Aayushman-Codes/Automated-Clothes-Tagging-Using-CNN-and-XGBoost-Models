import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { ArrowRight, Sparkles } from "lucide-react";
import Navigation from "@/components/Navigation";
import heroImage from "@/assets/hero-image.jpg";

const Home = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <main className="container mx-auto px-4 pt-32 pb-20">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div className="space-y-8 animate-fade-in">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-muted border border-border">
              <Sparkles className="w-4 h-4 text-primary" />
              <span className="text-sm font-medium text-muted-foreground">AI-Powered Fashion Intelligence</span>
            </div>
            
            <h1 className="text-5xl md:text-6xl font-bold leading-tight">
              Automated Product{" "}
              <span className="bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                Tagging System
              </span>
            </h1>
            
            <p className="text-xl text-muted-foreground leading-relaxed">
              AI that understands fashion. Upload clothing images and get instant, accurate tags and price predictions powered by advanced machine learning.
            </p>
            
            <div className="flex flex-wrap gap-4">
              <Button 
                asChild 
                size="lg" 
                className="text-lg px-8 py-6 shadow-soft hover:shadow-hover transition-all duration-300"
              >
                <Link to="/upload" className="flex items-center gap-2">
                  Try Now
                  <ArrowRight className="w-5 h-5" />
                </Link>
              </Button>
              
              <Button 
                asChild 
                variant="outline" 
                size="lg"
                className="text-lg px-8 py-6 transition-all duration-300"
              >
                <Link to="/learn-more">Learn More</Link>
              </Button>
            </div>
          </div>
          
          <div className="relative animate-scale-in">
            <div className="absolute inset-0 bg-gradient-hero opacity-20 blur-3xl rounded-full" />
            <img 
              src={heroImage}
              alt="AI Fashion Technology"
              className="relative rounded-2xl shadow-hover w-full"
            />
          </div>
        </div>
      </main>
    </div>
  );
};

export default Home;

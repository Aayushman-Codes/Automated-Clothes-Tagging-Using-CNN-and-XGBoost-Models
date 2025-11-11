import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Tag } from "lucide-react";

const Navigation = () => {
  const location = useLocation();
  
  const isActive = (path: string) => location.pathname === path;
  
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md border-b border-border">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-around max-w-3xl mx-auto">
          <Button
            variant={isActive("/") ? "default" : "ghost"}
            asChild
            className="transition-all duration-300"
          >
            <Link to="/">Home</Link>
          </Button>
          <Button
            variant={isActive("/upload") ? "default" : "ghost"}
            asChild
            className="transition-all duration-300"
          >
            <Link to="/upload">Upload</Link>
          </Button>
          <Button
            variant={isActive("/results") ? "default" : "ghost"}
            asChild
            className="transition-all duration-300"
          >
            <Link to="/results">Results</Link>
          </Button>
          <Button
            variant={isActive("/feedback") ? "default" : "ghost"}
            asChild
            className="transition-all duration-300"
          >
            <Link to="/feedback">Feedback</Link>
          </Button>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;

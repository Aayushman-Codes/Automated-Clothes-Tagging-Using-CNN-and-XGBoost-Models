import { Badge } from "@/components/ui/badge";

interface TagBubbleProps {
  tag: string;
  index: number;
}

const TagBubble = ({ tag, index }: TagBubbleProps) => {
  const colors = [
    "bg-blue-100 text-blue-700 border-blue-200",
    "bg-purple-100 text-purple-700 border-purple-200",
    "bg-teal-100 text-teal-700 border-teal-200",
    "bg-pink-100 text-pink-700 border-pink-200",
    "bg-indigo-100 text-indigo-700 border-indigo-200",
  ];
  
  const colorClass = colors[index % colors.length];
  
  return (
    <Badge 
      variant="outline" 
      className={`${colorClass} px-4 py-2 text-sm font-medium rounded-full border-2 animate-scale-in transition-all duration-300 hover:scale-110 hover:shadow-soft`}
      style={{ animationDelay: `${index * 0.1}s` }}
    >
      {tag}
    </Badge>
  );
};

export default TagBubble;

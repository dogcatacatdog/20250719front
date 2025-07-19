interface BlogPostProps {
  params: {
    slug: string;
  };
}

export default function BlogPost({ params }: BlogPostProps) {
  return (
    <div className="min-h-screen p-8">
      <h1 className="text-4xl font-bold mb-6">Blog Post: {params.slug}</h1>
      <p className="text-lg">
        This is a dynamic blog post page for slug: {params.slug}
      </p>
    </div>
  );
}

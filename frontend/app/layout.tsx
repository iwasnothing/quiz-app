import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "QuizGenius – AI-Powered Quiz Generator",
  description: "Define quiz DNA, generate questions, and refine them on a Gemini-style canvas.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="font-sans antialiased">
        {children}
      </body>
    </html>
  );
}

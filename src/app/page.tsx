"use client";

import { HeroSection } from "@/components/hero-section";
import { ProblemSection } from "@/components/problem-section";
import { SolutionSection } from "@/components/solution-section";
import { HowItWorksSection } from "@/components/how-it-works";
import { FeaturesSection } from "@/components/features-section";
import { ArchitectureSection } from "@/components/architecture-section";
import { CtaSection } from "@/components/cta-section";

export default function Home() {
  return (
    <div className="min-h-screen">
      <HeroSection />
      <ProblemSection />
      <SolutionSection />
      <HowItWorksSection />
      <FeaturesSection />
      <ArchitectureSection />
      <CtaSection />
      
      {/* Footer */}
      <footer className="py-8 text-center text-white/40 text-sm border-t border-white/5">
        <p>© {new Date().getFullYear()} TruthGuard AI. Demolition of Disinformation.</p>
      </footer>
    </div>
  );
}

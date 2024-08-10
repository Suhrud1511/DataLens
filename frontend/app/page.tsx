import CTA from "@/components/CTA";
import Faqs from "@/components/Faqs";
import Footer from "@/components/Footer";
import HeroSection from "@/components/HeroSection";
import Navbar from "@/components/Navbar";
import { ServiceSection } from "@/components/ServiceSection";

import Testimonials from "@/components/Testimonials";

export default function Home() {
  return (
    <>
      <div className="w-full dark:bg-black bg-white dark:bg-dot-white/[0.1] bg-dot-black/[0.1]">
        <Navbar />
        <HeroSection />
        <ServiceSection />
        <Testimonials />
        <Faqs />
        <CTA />
        <Footer />
      </div>
    </>
  );
}

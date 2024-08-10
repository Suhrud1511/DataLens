"use client";
import Image from "next/image";
import { FeaturesSection } from "./FeatureSections";
import { WobbleCard } from "./ui/wobble-card";

export function ServiceSection() {
  return (
    <section className="mx-auto w-full max-w-6xl px-6 py-10 sm:py-10 lg:px-8 lg:py-20 relative">
      <div className="mb-10">
        <h2 className="max-w-5xl mx-auto text-center tracking-tight font-medium text-black dark:text-white text-3xl md:text-5xl md:leading-tight">
          <span>Unleashing the Power of Data</span>
        </h2>
        <h2 className="text-sm md:text-base my-4 text-black/80 dark:text-white/80 font-normal dark:text-muted-dark text-center max-w-lg mx-auto">
          <span>
            Effortlessly preprocess and visualize your data with DataLens&apos;s
            intuitive and powerful features.
          </span>
        </h2>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 max-w-7xl mx-auto w-full">
        <WobbleCard
          containerClassName="col-span-1 lg:col-span-2 h-full bg-slate-800 min-h-[500px] lg:min-h-[300px]"
          className=""
        >
          <div className="max-w-sm">
            <h2 className="text-left text-balance text-base md:text-xl lg:text-3xl font-semibold tracking-[-0.015em] text-white">
              AutoPreprocessor: Simplify Data Processing
            </h2>
            <p className="mt-4 text-left text-base/6 text-neutral-200">
              Transform your CSV files with a single click. AutoPreprocessor
              streamlines data preparation, making it quick and easy to get your
              data analysis-ready.
            </p>
          </div>
          <Image
            src="/images/autopreprocessor-preview.png"
            width={500}
            height={500}
            alt="AutoPreprocessor demo image"
            className="absolute -right-4 lg:-right-[40%] grayscale filter -bottom-10 object-contain rounded-2xl"
          />
        </WobbleCard>

        <WobbleCard containerClassName="col-span-1 min-h-[300px] bg-slate-700">
          <h2 className="max-w-80 text-left text-balance text-base md:text-xl lg:text-3xl font-semibold tracking-[-0.015em] text-white">
            Advanced Data Visualization
          </h2>
          <p className="mt-4 max-w-[26rem] text-left text-base/6 text-neutral-200">
            Discover detailed insights with interactive data visualizations. Our
            platform helps you turn raw data into meaningful graphs and charts.
          </p>
        </WobbleCard>

        <WobbleCard containerClassName="col-span-1 lg:col-span-3 bg-slate-900 min-h-[500px] lg:min-h-[600px] xl:min-h-[300px]">
          <div className="max-w-sm">
            <h2 className="max-w-sm md:max-w-lg text-left text-balance text-base md:text-xl lg:text-3xl font-semibold tracking-[-0.015em] text-white">
              BirdEye: Uncover Insights Instantly
            </h2>
            <p className="mt-4 max-w-[26rem] text-left text-base/6 text-neutral-200">
              With BirdEye, upload your CSV and receive detailed visualizations
              and comprehensive reports in seconds. Get actionable insights
              quickly and efficiently.
            </p>
          </div>
          <Image
            src="/images/birdeye-preview.png"
            width={500}
            height={500}
            alt="BirdEye demo image"
            className="absolute -right-10 md:-right-[40%] lg:-right-[10%] -bottom-10 object-contain rounded-2xl"
          />
        </WobbleCard>
      </div>

      <FeaturesSection />
    </section>
  );
}

"use client";

import React from "react";
import { useMediaQuery } from "usehooks-ts";

import { testimonials } from "@/constants/data";
import ScrollingBanner from "./ui/scrolling-banner";
import UserReview from "./ui/user-review";

const Testimonials = () => {
  const testimonials1 = testimonials.slice(0, 4);
  const testimonials2 = testimonials.slice(4, 8);
  const testimonials3 = testimonials.slice(8, 12);
  const testimonials4 = testimonials.slice(12, 16);

  const isMobile = useMediaQuery("(max-width: 768px)");

  const fistColumn = React.useMemo(
    () => (isMobile ? testimonials : testimonials1),
    [isMobile, testimonials1]
  );

  return (
    <section className="mx-auto w-full max-w-6xl px-6 py-10 sm:py-32 lg:px-8 lg:py-10">
      <div className="mb-10">
        <h2 className="max-w-5xl mx-auto text-center tracking-tight font-medium text-black dark:text-white text-3xl md:text-5xl md:leading-tight">
          <span>Loved by people all over the universe</span>
        </h2>
        <h2 className="text-sm md:text-base my-4 text-black/80 dark:text-white/80 font-normal dark:text-muted-dark text-center max-w-lg mx-auto">
          <span>
            Join thousands who are transforming their data with DataLens.
            Experience the future of data analysis today.
          </span>
        </h2>
      </div>
      <div className="columns-1 sm:columns-2 md:columns-3 lg:columns-4">
        <ScrollingBanner
          isVertical
          duration={isMobile ? 200 : 120}
          shouldPauseOnHover={false}
        >
          {fistColumn.map((testimonial, index) => (
            <UserReview key={`${testimonial.name}-${index}`} {...testimonial} />
          ))}
        </ScrollingBanner>
        <ScrollingBanner
          isVertical
          className="hidden sm:flex"
          duration={200}
          shouldPauseOnHover={false}
        >
          {testimonials2.map((testimonial, index) => (
            <UserReview key={`${testimonial.name}-${index}`} {...testimonial} />
          ))}
        </ScrollingBanner>
        <ScrollingBanner
          isVertical
          className="hidden md:flex"
          duration={200}
          shouldPauseOnHover={false}
        >
          {testimonials3.map((testimonial, index) => (
            <UserReview key={`${testimonial.name}-${index}`} {...testimonial} />
          ))}
        </ScrollingBanner>
        <ScrollingBanner
          isVertical
          className="hidden lg:flex"
          duration={200}
          shouldPauseOnHover={false}
        >
          {testimonials4.map((testimonial, index) => (
            <UserReview key={`${testimonial.name}-${index}`} {...testimonial} />
          ))}
        </ScrollingBanner>
      </div>
    </section>
  );
};

export default Testimonials;

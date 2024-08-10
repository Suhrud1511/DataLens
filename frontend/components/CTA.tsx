"use client";
import Image from "next/image";
import Link from "next/link";
import { Button } from "./ui/button";

const CTA = () => {
  return (
    <div className="relative py-12 -mt-1 sm:py-24">
      <div className="relative w-[90%] mx-auto py-16 rounded-3xl z-10">
        <div className="flex bg-[url(/images/CardBG.png)] dark:opacity-90 bg-cover bg-no-repeat bg-center justify-between w-[90%] mx-auto rounded-3xl text-white py-16 shadow-md">
          <div className="flex flex-col items-center justify-center w-full gap-6">
            <h1 className="text-sm uppercase text-gray-50">
              Get started with DataLens for free
            </h1>
            <h2 className="w-[90%] text-2xl font-semibold text-center sm:w-2/4 sm:text-4xl">
              Ready to unlock the full potential of your data?
            </h2>
            <p className="w-[90%] sm:w-3/5 font-thin text-center">
              Join the DataLens community and experience seamless data
              preprocessing, insightful visualizations, and comprehensive
              reporting – all at your fingertips.
            </p>

            <div className="flex flex-col gap-4 sm:flex-row">
              <Button
                className="px-6 text-black bg-white hover:bg-slate-50 hover:text-black"
                onClick={() => {
                  console.log("Get Started with DataLens");
                }}
              >
                <Link href="/playground">Explore DataLens Now</Link>
              </Button>
            </div>

            <div className="flex flex-col items-center gap-4 text-sm text-gray-500 sm:-mt-3 sm:flex-row">
              <div className="flex gap-1">
                <Image
                  className="scale-90"
                  src="/images/card-Icon.png"
                  alt="card icon"
                  width={20}
                  height={20}
                />
                <p className="text-gray-400">No credit card required</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="h-96 w-96 rounded-full blur-3xl bg-[#ff5fe489]/10 absolute  top-52 -left-28" />
      <div className="h-96 w-96 rounded-full blur-3xl bg-[#d7b6f9]/20 absolute top-12 right-28" />
    </div>
  );
};

export default CTA;

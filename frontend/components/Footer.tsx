import { Chip } from "@nextui-org/react";
import { Separator } from "./ui/separator";

const Footer = () => {
  return (
    <footer className="flex w-full flex-col">
      <Separator className="bg-[#1e1d1d]" />
      <div className="mx-auto w-full max-w-7xl px-6 py-12 md:flex md:items-center md:justify-between lg:px-8">
        <div className="flex flex-col items-center justify-center gap-2 md:order-2 md:items-end"></div>
        <div className="mt-4 md:order-1 md:mt-0">
          <div className="flex items-center justify-center gap-3 md:justify-start">
            <div className="flex items-center">
              <span className="text-lg font-medium font-logo">datalens</span>
            </div>
            <Separator className="h-4" orientation="vertical" />
            <Chip
              className="border-none px-0 text-default-500 text-small"
              color="success"
              variant="dot"
            >
              All systems operational
            </Chip>
          </div>
          <p className="text-center text-default-400 md:text-start text-small">
            &copy; 2024 datalens. All rights reserved.
          </p>
        </div>
      </div>
      <p className="text-center text-5xl md:text-9xl lg:text-[18rem] font-bold bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 dark:from-neutral-950 to-neutral-200 dark:to-neutral-800 inset-x-0 font-logo">
        datalens
      </p>
    </footer>
  );
};

export default Footer;

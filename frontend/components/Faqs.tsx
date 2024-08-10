"use client";
import { faqs } from "@/constants/data";
import { Icon } from "@iconify/react";
import { Accordion, AccordionItem } from "@nextui-org/react";

const Faqs = () => {
  return (
    <section className="mx-auto w-full max-w-6xl py-20 sm:px-6 sm:py-32 lg:py-24">
      <div className="mx-auto flex w-full max-w-6xl flex-col items-center gap-6 lg:flex-row lg:items-start lg:gap-12">
        <h2 className="px-2 text-3xl leading-7">
          <span className="inline-block lg:hidden">FAQs</span>
        </h2>
        <h2 className="hidden bg-gradient-to-br from-foreground-800 to-foreground-500 bg-clip-text pt-4 text-5xl font-semibold tracking-tight text-transparent dark:to-foreground-200 lg:inline-block">
          Frequently
          <br />
          asked
          <br />
          questions
        </h2>
        <Accordion
          fullWidth
          keepContentMounted
          className="gap-3"
          itemClasses={{
            base: "px-0 sm:px-6",
            title: "font-medium",
            trigger: "py-6 flex-row-reverse",
            content: "pt-0 pb-6 text-base text-default-500",
          }}
          items={faqs}
          selectionMode="multiple"
        >
          {faqs?.map((item, i) => (
            <AccordionItem
              key={i}
              indicator={<Icon icon="lucide:plus" width={24} />}
              title={item.title}
            >
              {item.content}
            </AccordionItem>
          ))}
        </Accordion>
      </div>
    </section>
  );
};

export default Faqs;

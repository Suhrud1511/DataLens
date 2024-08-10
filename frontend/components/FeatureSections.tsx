import { cn } from "@/lib/utils";
import {
  IconAdjustmentsBolt,
  IconCloud,
  IconCurrencyDollar,
  IconEaseInOut,
  IconHeart,
  IconHelp,
  IconRouteAltLeft,
  IconTerminal2,
} from "@tabler/icons-react";

export function FeaturesSection() {
  const features = [
    {
      title: "Streamlined for Data Analysis",
      description:
        "Designed for data professionals, researchers, and anyone who wants to make sense of their data effortlessly.",
      icon: <IconTerminal2 />,
    },
    {
      title: "User-Friendly Interface",
      description:
        "Intuitive and easy-to-use, making data preprocessing and visualization a breeze.",
      icon: <IconEaseInOut />,
    },
    {
      title: "Completely Free",
      description:
        "No hidden fees or subscriptions. Enjoy all features at no cost with no credit card required.",
      icon: <IconCurrencyDollar />,
    },
    {
      title: "Reliable Performance",
      description:
        "We ensure maximum uptime, so you can trust us with your important data.",
      icon: <IconCloud />,
    },
    {
      title: "Flexible Access",
      description:
        "Easily manage your datasets and collaborate without restrictions.",
      icon: <IconRouteAltLeft />,
    },
    {
      title: "Support When You Need It",
      description:
        "Our dedicated support team is here to help you anytime you need assistance.",
      icon: <IconHelp />,
    },
    {
      title: "Satisfaction Guaranteed",
      description:
        "We’re committed to improving your data experience. Your feedback helps us grow.",
      icon: <IconAdjustmentsBolt />,
    },
    {
      title: "Continuous Updates",
      description:
        "We're always adding new features and improvements based on user feedback.",
      icon: <IconHeart />,
    },
  ];

  return (
    <section className="mx-auto w-full max-w-6xl px-6 sm:py-32 lg:px-8 lg:py-20">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4  relative z-10 py-20 max-w-7xl mx-auto">
        {features.map((feature, index) => (
          <Feature key={feature.title} {...feature} index={index} />
        ))}
      </div>
    </section>
  );
}

const Feature = ({
  title,
  description,
  icon,
  index,
}: {
  title: string;
  description: string;
  icon: React.ReactNode;
  index: number;
}) => {
  return (
    <div
      className={cn(
        "flex flex-col lg:border-r  py-10 relative group/feature dark:border-neutral-800",
        (index === 0 || index === 4) && "lg:border-l dark:border-neutral-800",
        index < 4 && "lg:border-b dark:border-neutral-800"
      )}
    >
      {index < 4 && (
        <div className="opacity-0 group-hover/feature:opacity-100 transition duration-200 absolute inset-0 h-full w-full bg-gradient-to-t from-neutral-100 dark:from-neutral-800 to-transparent pointer-events-none" />
      )}
      {index >= 4 && (
        <div className="opacity-0 group-hover/feature:opacity-100 transition duration-200 absolute inset-0 h-full w-full bg-gradient-to-b from-neutral-100 dark:from-neutral-800 to-transparent pointer-events-none" />
      )}
      <div className="mb-4 relative z-10 px-10 text-neutral-600 dark:text-neutral-400">
        {icon}
      </div>
      <div className="text-lg font-bold mb-2 relative z-10 px-10">
        <div className="absolute left-0 inset-y-0 h-6 group-hover/feature:h-8 w-1 rounded-tr-full rounded-br-full bg-neutral-300 dark:bg-neutral-700 group-hover/feature:bg-blue-500 transition-all duration-200 origin-center" />
        <span className="group-hover/feature:translate-x-2 transition duration-200 inline-block text-neutral-800 dark:text-neutral-100">
          {title}
        </span>
      </div>
      <p className="text-sm text-neutral-600 dark:text-neutral-300 max-w-xs relative z-10 px-10">
        {description}
      </p>
    </div>
  );
};

import AutoPreprocessorHowToUseCard from "@/components/card/AutopreprocessorHowToUseCard";
import BirdEyeHowToUseCard from "@/components/card/BirdEyeHowToUseCard";
import PageContainer from "@/components/layout/page-container";

const page = () => {
  return (
    <PageContainer scrollable={true}>
      <h2 className="text-2xl font-bold tracking-tight">Hi, Welcome back 👋</h2>

      <div className="mt-8 flex flex-wrap gap-6">
        <AutoPreprocessorHowToUseCard />
        <BirdEyeHowToUseCard />
      </div>
    </PageContainer>
  );
};

export default page;

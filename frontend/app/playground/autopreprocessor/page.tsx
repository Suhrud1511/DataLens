"use client";
import PageContainer from "@/components/layout/page-container";
import { Button } from "@/components/ui/button";
import { Heading } from "@/components/ui/heading";
import { Separator } from "@/components/ui/separator";
import { UploadButton } from "@/utils/uploadthing";
import { Plus } from "lucide-react";

const page = () => {
  return (
    <PageContainer>
      <div className="flex items-center justify-between mb-3">
        <Heading
          title="AutoPreprocessor"
          description="From Mess to Masterpiece: Preprocess your data in a jiffy 🎨"
        />

        <UploadButton
          appearance={{
            button:
              "ut-ready:bg-green-500 ut-uploading:cursor-not-allowed rounded-r-none bg-red-500 bg-none after:bg-orange-400",
            container: "w-max flex-row rounded-md border-cyan-300 bg-slate-800",
          }}
          endpoint="fileUploader"
          onClientUploadComplete={(res) => {
            console.log("Files: ", res);
            alert("Upload Completed");
          }}
          onUploadError={(error: Error) => {
            alert(`ERROR! ${error.message}`);
          }}
        />

        <Button
          className="text-xs md:text-sm"
          onClick={() => {
            console.log("button clicked");
          }}
        >
          <Plus className="mr-2 h-4 w-4" /> Add New
        </Button>
      </div>
      <Separator />

      <div className="flex flex-col gap-2 items-center justify-center h-96">
        <h2 className="text-2xl font-semibold tracking-tight">
          File Not Found
        </h2>
        <p className="text-sm text-slate-200">
          Please click the upload button in the top-right corner to upload your
          dataset.
        </p>
      </div>
    </PageContainer>
  );
};

export default page;

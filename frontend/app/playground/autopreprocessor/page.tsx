"use client";
import PageContainer from "@/components/layout/page-container";
import { Button } from "@/components/ui/button";
import { Heading } from "@/components/ui/heading";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { UserDatasetTable } from "@/components/UserDatasetTable";
import { UploadButton } from "@/utils/uploadthing";
import axios from "axios";
import { Plus } from "lucide-react";
import Papa from "papaparse";

import { useEffect, useState } from "react";

const Page = () => {
  const [datasetUrl, setDatasetUrl] = useState<string>("");
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [jsonData, setJsonData] = useState<any>(null);

  useEffect(() => {
    if (datasetUrl) {
      axios
        .get(datasetUrl, {
          responseType: "blob",
        })
        .then((response) => {
          const file = new Blob([response.data], { type: "text/csv" });
          setDatasetFile(file as File);

          const reader = new FileReader();
          reader.onload = (event) => {
            const text = event.target?.result as string;
            Papa.parse(text, {
              header: true,
              complete: (results) => {
                setJsonData(results.data);
              },
              error: () => {
                console.error("Error parsing CSV");
              },
            });
          };
          reader.readAsText(file);
        })
        .catch((error) => {
          console.error("Error fetching the CSV file: ", error);
        });
    }
  }, [datasetUrl]);

  const handleAutoPreprocessor = () => {
    console.log("Hello");
  };

  return (
    <PageContainer>
      <div className="flex items-center justify-between">
        <Heading
          title="AutoPreprocessor"
          description="From Mess to Masterpiece: Preprocess your data in a jiffy 🎨"
        />
        {!jsonData ? (
          <UploadButton
            appearance={{
              button:
                "bg-primary text-primary-foreground shadow hover:bg-primary/90 inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 px-4 w-26 mt-4 ut-button:bg-red-500 ut-button:ut-readying:bg-red-500/50",
              container: "text-xs md:text-sm text-slate-200",
              allowedContent: "",
            }}
            config={{ mode: "auto" }}
            content={{
              button({ ready }) {
                if (ready)
                  return (
                    <div className="flex items-center justify-center">
                      <Plus className="mr-1 h-4 w-4" /> Add Dataset
                    </div>
                  );

                return "Getting ready...";
              },
              allowedContent({ ready, fileTypes, isUploading }) {
                if (!ready) return "Checking what you allow";
                if (isUploading) return "Seems like dataset is uploading";
                return `You can upload CSV files`;
              },
            }}
            endpoint="fileUploader"
            onClientUploadComplete={(res) => {
              console.log("Files: ", res[0].url);
              setDatasetUrl(res[0].url);
            }}
            onUploadError={(error: Error) => {
              alert(`ERROR! ${error.message}`);
            }}
          />
        ) : (
          <Button onClick={handleAutoPreprocessor}>
            AutoPreprocess Dataset
          </Button>
        )}
      </div>
      <Separator className="my-3" />
      {jsonData ? (
        <Tabs defaultValue="user-dataset">
          <TabsList>
            <TabsTrigger value="user-dataset">User Dataset</TabsTrigger>
            <TabsTrigger value="preprocessed-dataset">
              Preprocessed Dataset
            </TabsTrigger>
          </TabsList>
          <TabsContent className="w-[90vw]" value="user-dataset">
            <UserDatasetTable jsonData={jsonData} />
          </TabsContent>
          <TabsContent value="preprocessed-dataset">TODO</TabsContent>
        </Tabs>
      ) : (
        <div className="flex flex-col gap-2 items-center justify-center h-96">
          <h2 className="text-2xl font-semibold tracking-tight">
            File Not Found
          </h2>
          <p className="text-sm text-slate-200">
            Please click the upload button in the top-right corner to upload
            your dataset.
          </p>
        </div>
      )}
    </PageContainer>
  );
};

export default Page;

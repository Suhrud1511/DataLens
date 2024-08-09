"use client";
import HeaderWithActionButton from "@/components/layout/HeaderWithActionButton";
import PageContainer from "@/components/layout/page-container";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { UserDatasetTable } from "@/components/UserDatasetTable";
import axios from "axios";
import Papa from "papaparse";
import { useEffect, useState } from "react";

const Birdeye = () => {
  const [datasetUrl, setDatasetUrl] = useState<string>(() => {
    return localStorage.getItem("datasetUrl") || "";
  });
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [jsonData, setJsonData] = useState<any>(null);
  const [preprocessedData, setPreprocessedData] = useState<File | null>(null);
  const [preprocessedJsonData, setPreprocessedJsonData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    if (datasetUrl) {
      setLoading(true);
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
          setLoading(false);
        })
        .catch((error) => {
          console.error("Error fetching the CSV file: ", error);
        });
    }
  }, [datasetUrl]);

  return (
    <PageContainer>
      <HeaderWithActionButton
        setDatasetUrl={setDatasetUrl}
        title="Bird Eye"
        description="Uncover the Details: Visualize and Report Your Data with Precision 🔍"
      />
      <Separator className="my-4" />
      {jsonData ? (
        <Tabs defaultValue="user-dataset">
          <div className="flex justify-between">
            <TabsList className="mb-2">
              <TabsTrigger value="user-dataset">User Dataset</TabsTrigger>
              <TabsTrigger value="generate-report">Dataset Report</TabsTrigger>
            </TabsList>
            {loading ? (
              <Button disabled>Generating Report</Button>
            ) : preprocessedData ? (
              <Button>Download Generated Report</Button>
            ) : (
              <Button onClick={() => console.log("first")}>
                Generate Report
              </Button>
            )}
          </div>
          <TabsContent className="w-[90vw]" value="user-dataset">
            <UserDatasetTable jsonData={jsonData} />
          </TabsContent>
          <TabsContent value="generate-report">
            {preprocessedData ? (
              <h1>TODO</h1>
            ) : (
              <p>
                No report available. Please generate a report for your dataset.
              </p>
            )}
          </TabsContent>
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

export default Birdeye;

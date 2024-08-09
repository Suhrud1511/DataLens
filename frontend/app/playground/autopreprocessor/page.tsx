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

const Autopreprocessor = () => {
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

  const handleAutoPreprocessor = async () => {
    if (datasetUrl) {
      setLoading(true);
      try {
        const response = await axios.get("http://127.0.0.1:5000/preprocess", {
          params: {
            url: datasetUrl,
          },
          responseType: "blob",
        });
        const preprocessedFile = response.data;

        // Convert Blob to a file-like object for further handling
        const file = new File([preprocessedFile], "preprocessed_data.csv", {
          type: "text/csv",
        });

        setPreprocessedData(file);

        // Parse the CSV file to JSON
        const reader = new FileReader();
        reader.onload = (event) => {
          const text = event.target?.result as string;
          Papa.parse(text, {
            header: true,
            complete: (results) => {
              setPreprocessedJsonData(results.data);
            },
            error: () => {
              console.error("Error parsing CSV");
            },
          });
        };
        reader.readAsText(file);
        setLoading(false);
      } catch (error: any) {
        console.error("Error during preprocessing: ", error);
        alert("Error during preprocessing: " + error.message);
      }
    } else {
      alert("No dataset URL found. Please upload a dataset first.");
    }
  };

  const handleDownloadPreprocessedData = async () => {
    if (preprocessedData) {
      const url = window.URL.createObjectURL(preprocessedData);
      const a = document.createElement("a");
      a.style.display = "none";
      a.href = url;
      a.download = "preprocessed_data.csv";
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
    } else {
      alert(
        "No preprocessed data available. Please preprocess the dataset first."
      );
    }
  };

  return (
    <PageContainer>
      <HeaderWithActionButton
        setDatasetUrl={setDatasetUrl}
        title="AutoPreprocessor"
        description="From Mess to Masterpiece: Preprocess your data in a jiffy 🎨"
      />
      <Separator className="my-4" />
      {jsonData ? (
        <Tabs defaultValue="user-dataset">
          <div className="flex justify-between">
            <TabsList className="mb-2">
              <TabsTrigger value="user-dataset">User Dataset</TabsTrigger>
              <TabsTrigger value="preprocessed-dataset">
                Preprocessed Dataset
              </TabsTrigger>
            </TabsList>
            {loading ? (
              <Button disabled>AutoPreprocessing Dataset</Button>
            ) : preprocessedData ? (
              <Button onClick={handleDownloadPreprocessedData}>
                Download Preprocessed Dataset
              </Button>
            ) : (
              <Button onClick={handleAutoPreprocessor}>
                AutoPreprocess Dataset
              </Button>
            )}
          </div>
          <TabsContent className="w-[90vw]" value="user-dataset">
            <UserDatasetTable jsonData={jsonData} />
          </TabsContent>
          <TabsContent value="preprocessed-dataset">
            {preprocessedData ? (
              <UserDatasetTable jsonData={preprocessedJsonData} />
            ) : (
              <p>
                No preprocessed data available. Please preprocess the dataset.
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

export default Autopreprocessor;

import { Heading } from "@/components/ui/heading";
import { UploadButton } from "@/utils/uploadthing";
import { Plus } from "lucide-react";

interface HeaderWithActionButtonProps {
  setDatasetUrl: (url: string) => void;
  title: string;
  description: string;
}

const HeaderWithActionButton = ({
  setDatasetUrl,
  title,
  description,
}: HeaderWithActionButtonProps) => {
  const handleUploadComplete = (res: any) => {
    const url = res[0].url;
    setDatasetUrl(url);
    localStorage.setItem("datasetUrl", url);
  };

  return (
    <div className="flex items-center justify-between">
      <Heading title={title} description={description} />

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
        onClientUploadComplete={handleUploadComplete}
        onUploadError={(error: Error) => {
          alert(`ERROR! ${error.message}`);
        }}
      />
    </div>
  );
};

export default HeaderWithActionButton;

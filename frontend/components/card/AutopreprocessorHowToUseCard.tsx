import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import Link from "next/link";
import { Button } from "../ui/button";

const AutoPreprocessorHowToUseCard = () => {
  return (
    <Card className="max-w-lg">
      <CardHeader className="text-medium font-semibold">
        <CardTitle>AutoPreprocessor - How to Use</CardTitle>
        <CardDescription className="text-xs">
          Follow these steps to preprocess your dataset efficiently.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <div>
          <h3 className="font-semibold">1. Upload Your Dataset</h3>
          <p>
            Click on the <strong>&quot;Add Dataset&quot;</strong> button in the
            top-right corner. Once uploaded, a preview of your dataset will be
            displayed below.
          </p>
        </div>
        <div>
          <h3 className="font-semibold">2. Autopreprocess Your Dataset</h3>
          <p>
            Hit the <strong>&quot;Autopreprocess Dataset&quot;</strong>
            button on the top-right. The process might take a few seconds,
            depending on the dataset&apos;s size.
          </p>
        </div>
        <div>
          <h3 className="font-semibold">3. View Preprocessed Dataset</h3>
          <p>
            Switch to the <strong>&quot;Preprocessed Dataset&quot;</strong>
            tab to see a preview of your cleaned and transformed data.
          </p>
        </div>
        <div>
          <h3 className="font-semibold">4. Download Preprocessed Data</h3>
          <p>
            Download the processed dataset by clicking the
            <strong>&quot;Download Preprocessed Dataset&quot;</strong>
            button.
          </p>
        </div>
      </CardContent>
      <CardFooter>
        <Button className="text-sm">
          <Link href="/playground/autopreprocessor">
            Start preprocessing your data today!
          </Link>
        </Button>
      </CardFooter>
    </Card>
  );
};

export default AutoPreprocessorHowToUseCard;

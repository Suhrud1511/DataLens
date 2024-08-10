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

export default function BirdEyeHowToUseCard() {
  return (
    <Card className="max-w-lg">
      <CardHeader className="text-medium font-semibold">
        <CardTitle>Bird Eye - How to Use</CardTitle>
        <CardDescription className="text-xs">
          Follow these steps to visualize and report your dataset with
          precision.
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
          <h3 className="font-semibold">2. Generate Your Report</h3>
          <p>
            Click the <strong>&quot;Generate Report&quot;</strong> button on the
            top-right. The report generation might take a few minutes, depending
            on the size of the dataset.
          </p>
        </div>
        <div>
          <h3 className="font-semibold">3. View the Dataset Report</h3>
          <p>
            Switch to the <strong>&quot;Dataset Report&quot;</strong> tab to see
            a preview of the generated report for your dataset.
          </p>
        </div>
        <div>
          <h3 className="font-semibold">4. Download the Report</h3>
          <p>
            Download the generated report by clicking the
            <strong>&quot;Download Generated Report&quot;</strong> button.
          </p>
        </div>
      </CardContent>
      <CardFooter>
        <Button className="text-sm">
          <Link href="/playground/birdeye">
            Visualize your data with Bird Eye now!
          </Link>
        </Button>
      </CardFooter>
    </Card>
  );
}

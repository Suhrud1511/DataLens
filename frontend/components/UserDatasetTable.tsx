import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ScrollArea, ScrollBar } from "./ui/scroll-area";

export function UserDatasetTable({ jsonData }: { jsonData: any }) {
  if (!jsonData || jsonData.length === 0) {
    return <div>No data available</div>;
  }

  const headers = Object.keys(jsonData[0]);

  return (
    <ScrollArea className="h-[calc(80vh-180px)] max-w-[92vw] sm:max-w-[76vw] overflow-auto w-max rounded-md border md:h-[calc(80dvh-120px)]">
      <Table className="w-max">
        <TableCaption>User Uploaded Dataset</TableCaption>
        <TableHeader className="sticky top-0 bg-secondary">
          <TableRow>
            {headers.map((header) => (
              <TableHead key={header}>{header}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {jsonData.map((dataPoint: any, index: number) => (
            <TableRow key={index}>
              {headers.map((header) => (
                <TableCell key={header}>{dataPoint[header]}</TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <ScrollBar orientation="horizontal" />
    </ScrollArea>
  );
}

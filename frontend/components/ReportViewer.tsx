import DOMPurify from "dompurify";
import PageContainer from "./layout/page-container";

const ReportViewer = ({ reportHTML }: { reportHTML: string }) => {
  const sanitizedContent = DOMPurify.sanitize(reportHTML);

  return (
    <PageContainer scrollable={true}>
      <div dangerouslySetInnerHTML={{ __html: sanitizedContent }} />
    </PageContainer>
  );
};

export default ReportViewer;

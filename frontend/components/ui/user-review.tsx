"use client";

import { cn } from "@/lib/utils";
import { Avatar } from "@nextui-org/react";
import React from "react";

export type UserReviewProps = React.HTMLAttributes<HTMLDivElement> & {
  avatar: string;
  name: string;
  role: string;
  content: string;
};

const UserReview = React.forwardRef<HTMLDivElement, UserReviewProps>(
  ({ children, name, avatar, content, className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "flex flex-col gap-2.5 rounded-medium bg-content1 p-5 shadow-small",
        className
      )}
      {...props}
    >
      <div className="flex items-center gap-2">
        <Avatar alt={name} className="h-7 w-7" size="sm" src={avatar} />
        <span className="text-small text-foreground">{name}</span>
      </div>
      <p className="text-default-700">{content || children}</p>
    </div>
  )
);

UserReview.displayName = "UserReview";

export default UserReview;

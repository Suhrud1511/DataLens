import Link from "next/link";

const Navbar = () => {
  return (
    <nav className="flex justify-between px-16 py-8 items-center">
      <h1 className="font-semibold text-3xl text-slate-200 font-logo z-10">
        datalens
      </h1>

      <ul className="flex gap-6 items-center font-semibold text-slate-200 font-logo z-10">
        <li>
          <Link href="/">Feedback</Link>
        </li>
        <li>
          <Link href="/playground">Playground</Link>
        </li>
      </ul>
    </nav>
  );
};

export default Navbar;

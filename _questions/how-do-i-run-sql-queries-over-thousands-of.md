---
title: "How do I run SQL queries over thousands of similar PDFs, such as reports or forms that follow one template?"
section: database-systems
order: 13
description: "We recommend defining a table over the documents and extracting only the fields a query needs, using the shared template to find them. Our ZenDB system (ICDE 2025) detects the common section structure of templatized documents, such as civic agenda reports, as a tree for each document. You then query the collection with SQL, and ZenDB sends the LLM only the sections that matter."
guides: [document-processing]
sources:
  - title: "Querying Templatized Document Collections with Large Language Models (ICDE 2025)"
    url: https://www.computer.org/csdl/proceedings-article/icde/2025/360300c422/26FZAUkb1ra
asked:
  - https://reddit.com/r/Rag/comments/1sg1luf/is_rag_what_i_should_be_using/
  - https://bittide.aicompass.dev/article/92d4bf13-ea82-4a4c-be5d-99e353093d28
---

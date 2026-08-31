-- Adds "personal" as a valid document category, for real personal documents
-- (bills, insurance policies, e-tickets -- see data/raw_docs/personal/) as
-- opposed to the synthetic company HR/IT/travel policy docs. Kept as a
-- separate migration rather than editing 002_documents.sql so the history
-- of schema changes stays honest (that file already shipped/ran).
ALTER TABLE documents DROP CONSTRAINT documents_category_check;
ALTER TABLE documents ADD CONSTRAINT documents_category_check
    CHECK (category IN ('hr', 'it', 'travel', 'personal'));

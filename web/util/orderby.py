from django.db import backend, connection, models
from django.utils.datastructures import SortedDict

# UGH, this is a major hack because the Django code is too rigid to be
# subclassed easily.  Much cut-n-pasting from django/db/models/query.py .

class OrderByManager(models.Manager):
    def get_query_set(self):
        return OrderByQuerySet(self.model)

def handle_legacy_orderlist(order_list):
    return models.query.handle_legacy_orderlist(order_list)

class OrderByQuerySet(models.query.QuerySet):
    _orderby_sql = None

    def order_by_expression(self, sql):
        return self._clone(_orderby_sql = sql)

    # This is the function the superclass should have defined...
    def _super_get_sql_orderby_clause(self, opts):
        order_by = []
        if self._order_by is not None:
            ordering_to_use = self._order_by
        else:
            ordering_to_use = opts.ordering
        for f in handle_legacy_orderlist(ordering_to_use):
            if f == '?': # Special case.
                order_by.append(connection.ops.random_function_sql())
            else:
                if f.startswith('-'):
                    col_name = f[1:]
                    order = "DESC"
                else:
                    col_name = f
                    order = "ASC"
                if "." in col_name:
                    table_prefix, col_name = col_name.split('.', 1)
                    table_prefix = connection.ops.quote_name(table_prefix) + '.'
                else:
                    # Use the database table as a column prefix if it wasn't given,
                    # and if the requested column isn't a custom SELECT.
                    if "." not in col_name and col_name not in (self._select or ()):
                        table_prefix = connection.ops.quote_name(opts.db_table) + '.'
                    else:
                        table_prefix = ''
                    order_by.append('%s%s %s' % (table_prefix, connection.ops.quote_name(orderfield2column(col_name, opts)), order))
        if order_by:
            return "ORDER BY " + ", ".join(order_by)
        return None

    # This is how we would have overridden the superclass's implementation...
    def _get_sql_orderby_clause(self, opts):
        if self._orderby_sql:
            return ' ORDER BY ' + self._orderby_sql
        return self._super_get_sql_orderby_clause(opts)

    # This is copied from the superclass because it's a monolithic chunk of
    # code that doesn't allow the parts to be overridden...
    def _get_sql_clause(self):
        opts = self.model._meta

        # Construct the fundamental parts of the query: SELECT X FROM Y WHERE Z.
        select = ["%s.%s" % (connection.ops.quote_name(opts.db_table), connection.ops.quote_name(f.column)) for f in opts.fields]
        tables = [quote_only_if_word(t) for t in self._tables]
        joins = SortedDict()
        where = self._where[:]
        params = self._params[:]

        # Convert self._filters into SQL.
        joins2, where2, params2 = self._filters.get_sql(opts)
        joins.update(joins2)
        where.extend(where2)
        params.extend(params2)

        # Add additional tables and WHERE clauses based on select_related.
        if self._select_related:
            fill_table_cache(opts, select, tables, where, 
                             old_prefix=opts.db_table, 
                             cache_tables_seen=[opts.db_table], 
                             max_depth=self._max_related_depth)

        # Add any additional SELECTs.
        if self._select:
            select.extend(['(%s) AS %s' % (quote_only_if_word(s[1]), connection.ops.quote_name(s[0])) for s in self._select.items()])

        # Start composing the body of the SQL statement.
        sql = [" FROM", connection.ops.quote_name(opts.db_table)]

        # Compose the join dictionary into SQL describing the joins.
        if joins:
            sql.append(" ".join(["%s %s AS %s ON %s" % (join_type, table, alias, condition)
                                 for (alias, (table, join_type, condition)) in joins.items()]))

        # Compose the tables clause into SQL.
        if tables:
            sql.append(", " + ", ".join(tables))

        # Compose the where clause into SQL.
        if where:
            sql.append(where and "WHERE " + " AND ".join(where))

        # ORDER BY clause
        orderby_clause = self._get_sql_orderby_clause(opts)
        if orderby_clause:
            sql.append(orderby_clause)

        # LIMIT and OFFSET clauses
        if self._limit is not None:
            sql.append("%s " % connection.ops.limit_offset_sql(self._limit, self._offset))
        else:
            assert self._offset is None, "'offset' is not allowed without 'limit'"
        #logging.debug("Returning SQL: " + " ".join(sql))
        return select, " ".join(sql), params

    def __str__(self):
        return '<OrderByQuerySet: _orderby_sql=' + (self._orderby_sql or "none") + '>'

    def _clone(self, klass=None, **kwargs):
        c = super(OrderByQuerySet, self)._clone(klass, **kwargs)
        c._orderby_sql = self._orderby_sql
        # Not sure why this doesn't work... it's straight from query.py
        #c.__dict__.update(kwargs)
        if '_orderby_sql' in kwargs:
            c._orderby_sql = kwargs['_orderby_sql']
        return c


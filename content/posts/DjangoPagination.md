---
title: "Use Django for pagination"
date: 2023-09-28
draft: false

tags: ["python"]
categories: ["backend"]
---



### Create Pageinator

```python
class FantasyPagination(PageNumberPagination):
    page_size = 5
    page_size_query_param = 'page_size'
    page_query_param = 'page'
    max_page_size = 10
```

### Use

```python
 records = FantasyRecord.objects.all().order_by("-data")
 paginator = FantasyPagination()
 page = paginator.paginate_queryset(records, request)
 serializer = FantasyRecordSerializer(page, many=True)
```


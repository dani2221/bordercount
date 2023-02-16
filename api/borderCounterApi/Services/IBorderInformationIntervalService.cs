using System;
using borderCounterApi.DTOs.BorderInformationInterval;
using borderCounterApi.Models;

namespace borderCounterApi.Services
{
	public interface IBorderInformationIntervalService
	{
		Task<Guid> Add(BorderInformationIntervalRequest interval);
		Task<BorderInformationIntervalResponse> Get(Guid guid);
		Task<List<BorderInformationIntervalResponse>> GetLast(string border, int limit, int skipAggregate);
	}
}

